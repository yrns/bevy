use super::{
    state_descriptors::PrimitiveTopology, ComputePipelineDescriptor, Descriptor, IndexFormat,
    PipelineDescriptor, VertexBufferDescriptor,
};
use crate::{
    pipeline::BindType,
    renderer::RenderResourceContext,
    shader::{Shader, ShaderError, ShaderSource},
};
use bevy_asset::{Assets, Handle};
use bevy_reflect::Reflect;
use bevy_utils::{HashMap, HashSet};
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};

pub trait PipelineSpecialization: std::fmt::Debug + Clone + Default + Reflect + Eq {
    fn get_shader_specialization(&self) -> &ShaderSpecialization;
    fn get_dynamic_bindings(&self) -> &HashSet<String>;
}

macro_rules! impl_pipeline_specialization {
    ($ty:ident) => {
        impl PipelineSpecialization for $ty {
            fn get_shader_specialization(&self) -> &ShaderSpecialization {
                &self.shader_specialization
            }

            fn get_dynamic_bindings(&self) -> &HashSet<String> {
                &self.dynamic_bindings
            }
        }
    };
}

#[derive(Clone, Eq, PartialEq, Debug, Reflect)]
pub struct RenderPipelineSpecialization {
    pub shader_specialization: ShaderSpecialization,
    pub primitive_topology: PrimitiveTopology,
    pub dynamic_bindings: HashSet<String>,
    pub index_format: IndexFormat,
    pub vertex_buffer_descriptor: VertexBufferDescriptor,
    pub sample_count: u32,
}

impl Default for RenderPipelineSpecialization {
    fn default() -> Self {
        Self {
            sample_count: 1,
            index_format: IndexFormat::Uint32,
            shader_specialization: Default::default(),
            primitive_topology: Default::default(),
            dynamic_bindings: Default::default(),
            vertex_buffer_descriptor: Default::default(),
        }
    }
}

impl RenderPipelineSpecialization {
    pub fn empty() -> &'static RenderPipelineSpecialization {
        pub static EMPTY: Lazy<RenderPipelineSpecialization> =
            Lazy::new(RenderPipelineSpecialization::default);
        &EMPTY
    }
}

impl_pipeline_specialization!(RenderPipelineSpecialization);

#[derive(Clone, Eq, PartialEq, Debug, Default, Reflect, Serialize, Deserialize)]
pub struct ShaderSpecialization {
    pub shader_defs: HashSet<String>,
}

#[derive(Debug)]
pub(crate) struct SpecializedShader {
    pub(crate) shader: Handle<Shader>,
    pub(crate) specialization: ShaderSpecialization,
}

#[derive(Debug)]
struct SpecializedPipeline<T: Descriptor> {
    pipeline: Handle<T>,
    specialization: T::Specialization,
}

#[derive(Clone, Eq, PartialEq, Debug, Reflect, Default)]
pub struct ComputePipelineSpecialization {
    pub shader_specialization: ShaderSpecialization,
    pub dynamic_bindings: HashSet<String>,
}

impl_pipeline_specialization!(ComputePipelineSpecialization);

pub type RenderPipelineCompiler = PipelineCompiler<PipelineDescriptor>;
pub type ComputePipelineCompiler = PipelineCompiler<ComputePipelineDescriptor>;

#[derive(Debug)]
pub struct PipelineCompiler<T: Descriptor> {
    specialized_shaders: HashMap<Handle<Shader>, Vec<SpecializedShader>>,
    specialized_shader_pipelines: HashMap<Handle<Shader>, Vec<Handle<T>>>,
    specialized_pipelines: HashMap<Handle<T>, Vec<SpecializedPipeline<T>>>,
}

// Like Pipeline, this needs a manual impl.
impl<T: Descriptor> Default for PipelineCompiler<T> {
    fn default() -> Self {
        Self {
            specialized_shaders: Default::default(),
            specialized_shader_pipelines: Default::default(),
            specialized_pipelines: Default::default(),
        }
    }
}

impl<T: Descriptor> PipelineCompiler<T> {
    fn compile_shader(
        &mut self,
        render_resource_context: &dyn RenderResourceContext,
        shaders: &mut Assets<Shader>,
        shader_handle: &Handle<Shader>,
        shader_specialization: &ShaderSpecialization,
    ) -> Result<Handle<Shader>, ShaderError> {
        let specialized_shaders = self
            .specialized_shaders
            .entry(shader_handle.clone_weak())
            .or_insert_with(Vec::new);

        let shader = shaders.get(shader_handle).unwrap();

        // don't produce new shader if the input source is already spirv
        if let ShaderSource::Spirv(_) = shader.source {
            return Ok(shader_handle.clone_weak());
        }

        if let Some(specialized_shader) =
            specialized_shaders
                .iter()
                .find(|current_specialized_shader| {
                    current_specialized_shader.specialization == *shader_specialization
                })
        {
            // if shader has already been compiled with current configuration, use existing shader
            Ok(specialized_shader.shader.clone_weak())
        } else {
            // if no shader exists with the current configuration, create new shader and compile
            let shader_def_vec = shader_specialization
                .shader_defs
                .iter()
                .cloned()
                .collect::<Vec<String>>();
            let compiled_shader =
                render_resource_context.get_specialized_shader(shader, Some(&shader_def_vec))?;
            let specialized_handle = shaders.add(compiled_shader);
            let weak_specialized_handle = specialized_handle.clone_weak();
            specialized_shaders.push(SpecializedShader {
                shader: specialized_handle,
                specialization: shader_specialization.clone(),
            });
            Ok(weak_specialized_handle)
        }
    }

    pub fn get_specialized_pipeline(
        &self,
        pipeline: &Handle<T>,
        specialization: &T::Specialization,
    ) -> Option<Handle<T>> {
        self.specialized_pipelines
            .get(pipeline)
            .and_then(|specialized_pipelines| {
                specialized_pipelines
                    .iter()
                    .find(|current_specialized_pipeline| {
                        &current_specialized_pipeline.specialization == specialization
                    })
            })
            .map(|specialized_pipeline| specialized_pipeline.pipeline.clone_weak())
    }

    pub fn compile_pipeline(
        &mut self,
        render_resource_context: &dyn RenderResourceContext,
        pipelines: &mut Assets<T>,
        shaders: &mut Assets<Shader>,
        source_pipeline: &Handle<T>,
        pipeline_specialization: &T::Specialization,
    ) -> Handle<T> {
        let source_descriptor = pipelines.get(source_pipeline).unwrap();

        let shader_stages = source_descriptor.get_shader_stages().map(|s| {
            self.compile_shader(
                render_resource_context,
                shaders,
                s,
                &pipeline_specialization.get_shader_specialization(),
            )
            .unwrap()
        });

        let mut layout =
            render_resource_context.reflect_pipeline_layout(&shaders, &shader_stages, true);

        if !pipeline_specialization.get_dynamic_bindings().is_empty() {
            // set binding uniforms to dynamic if render resource bindings use dynamic
            for bind_group in layout.bind_groups.iter_mut() {
                let mut binding_changed = false;
                for binding in bind_group.bindings.iter_mut() {
                    if pipeline_specialization
                        .get_dynamic_bindings()
                        .iter()
                        .any(|b| b == &binding.name)
                    {
                        if let BindType::Uniform {
                            ref mut dynamic, ..
                        } = binding.bind_type
                        {
                            *dynamic = true;
                            binding_changed = true;
                        }
                    }
                }

                if binding_changed {
                    bind_group.update_id();
                }
            }
        }

        // Create a specialized pipeline from the source pipeline.
        let specialized_descriptor =
            source_descriptor.specialize_from(pipeline_specialization, shader_stages, layout);

        // track specialized shader pipelines
        for s in specialized_descriptor.get_shader_stages().iter() {
            self.specialized_shader_pipelines
                .entry(s.clone_weak())
                .or_insert_with(Default::default)
                .push(source_pipeline.clone_weak());
        }

        let specialized_pipeline_handle = pipelines.add(specialized_descriptor);
        pipelines
            .get(&specialized_pipeline_handle)
            .unwrap()
            .create_pipeline(
                render_resource_context,
                specialized_pipeline_handle.clone_weak(),
                &shaders,
            );

        let specialized_pipelines = self
            .specialized_pipelines
            .entry(source_pipeline.clone_weak())
            .or_insert_with(Vec::new);
        let weak_specialized_pipeline_handle = specialized_pipeline_handle.clone_weak();
        specialized_pipelines.push(SpecializedPipeline {
            pipeline: specialized_pipeline_handle,
            specialization: pipeline_specialization.clone(),
        });

        weak_specialized_pipeline_handle
    }

    pub fn iter_compiled_pipelines(
        &self,
        pipeline_handle: Handle<T>,
    ) -> Option<impl Iterator<Item = &Handle<T>>> {
        if let Some(compiled_pipelines) = self.specialized_pipelines.get(&pipeline_handle) {
            Some(
                compiled_pipelines
                    .iter()
                    .map(|specialized_pipeline| &specialized_pipeline.pipeline),
            )
        } else {
            None
        }
    }

    pub fn iter_all_compiled_pipelines(&self) -> impl Iterator<Item = &Handle<T>> {
        self.specialized_pipelines
            .values()
            .map(|compiled_pipelines| {
                compiled_pipelines
                    .iter()
                    .map(|specialized_pipeline| &specialized_pipeline.pipeline)
            })
            .flatten()
    }

    /// Update specialized shaders and remove any related specialized
    /// pipelines and assets.
    pub fn update_shader(
        &mut self,
        shader: &Handle<Shader>,
        pipelines: &mut Assets<PipelineDescriptor>,
        shaders: &mut Assets<Shader>,
        render_resource_context: &dyn RenderResourceContext,
    ) -> Result<(), ShaderError> {
        if let Some(specialized_shaders) = self.specialized_shaders.get_mut(shader) {
            for specialized_shader in specialized_shaders {
                // Recompile specialized shader. If it fails, we bail immediately.
                let shader_def_vec = specialized_shader
                    .specialization
                    .shader_defs
                    .iter()
                    .cloned()
                    .collect::<Vec<String>>();
                let new_handle =
                    shaders.add(render_resource_context.get_specialized_shader(
                        shaders.get(shader).unwrap(),
                        Some(&shader_def_vec),
                    )?);

                // Replace handle and remove old from assets.
                let old_handle = std::mem::replace(&mut specialized_shader.shader, new_handle);
                shaders.remove(&old_handle);

                // Find source pipelines that use the old specialized
                // shader, and remove from tracking.
                if let Some(source_pipelines) =
                    self.specialized_shader_pipelines.remove(&old_handle)
                {
                    // Remove all specialized pipelines from tracking
                    // and asset storage. They will be rebuilt on next
                    // draw.
                    for source_pipeline in source_pipelines {
                        if let Some(specialized_pipelines) =
                            self.specialized_pipelines.remove(&source_pipeline)
                        {
                            for p in specialized_pipelines {
                                pipelines.remove(p.pipeline);
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }
}
