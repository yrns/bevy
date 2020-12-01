use super::PipelineLayout;
use super::{ComputePipelineDescriptor, ShaderSpecialization, SpecializedShader};
use crate::{
    pipeline::BindType,
    renderer::RenderResourceContext,
    shader::{Shader, ShaderSource},
};
use bevy_asset::{Assets, Handle};
use bevy_reflect::Reflect;
use bevy_utils::{HashMap, HashSet};
use once_cell::sync::Lazy;

#[derive(Clone, Eq, PartialEq, Debug, Reflect)]
pub struct ComputePipelineSpecialization {
    pub shader_specialization: ShaderSpecialization,
    pub dynamic_bindings: HashSet<String>,
}

impl Default for ComputePipelineSpecialization {
    fn default() -> Self {
        Self {
            shader_specialization: Default::default(),
            dynamic_bindings: Default::default(),
        }
    }
}

impl ComputePipelineSpecialization {
    pub fn empty() -> &'static ComputePipelineSpecialization {
        pub static EMPTY: Lazy<ComputePipelineSpecialization> =
            Lazy::new(ComputePipelineSpecialization::default);
        &EMPTY
    }
}

struct ComputeSpecializedPipeline {
    pipeline: Handle<ComputePipelineDescriptor>,
    specialization: ComputePipelineSpecialization,
}

#[derive(Default)]
pub struct ComputePipelineCompiler {
    specialized_shaders: HashMap<Handle<Shader>, Vec<SpecializedShader>>,
    specialized_pipelines:
        HashMap<Handle<ComputePipelineDescriptor>, Vec<ComputeSpecializedPipeline>>,
}

impl ComputePipelineCompiler {
    // TODO: Share some of this with PipelineCompiler.
    fn compile_shader(
        &mut self,
        shaders: &mut Assets<Shader>,
        shader_handle: &Handle<Shader>,
        shader_specialization: &ShaderSpecialization,
    ) -> Handle<Shader> {
        let specialized_shaders = self
            .specialized_shaders
            .entry(shader_handle.clone_weak())
            .or_insert_with(Vec::new);

        let shader = shaders.get(shader_handle).unwrap();

        // don't produce new shader if the input source is already spirv
        if let ShaderSource::Spirv(_) = shader.source {
            return shader_handle.clone_weak();
        }

        if let Some(specialized_shader) =
            specialized_shaders
                .iter()
                .find(|current_specialized_shader| {
                    current_specialized_shader.specialization == *shader_specialization
                })
        {
            // if shader has already been compiled with current configuration, use existing shader
            specialized_shader.shader.clone_weak()
        } else {
            // if no shader exists with the current configuration, create new shader and compile
            let shader_def_vec = shader_specialization
                .shader_defs
                .iter()
                .cloned()
                .collect::<Vec<String>>();
            let compiled_shader = shader.get_spirv_shader(Some(&shader_def_vec));
            let specialized_handle = shaders.add(compiled_shader);
            let weak_specialized_handle = specialized_handle.clone_weak();
            specialized_shaders.push(SpecializedShader {
                shader: specialized_handle,
                specialization: shader_specialization.clone(),
            });
            weak_specialized_handle
        }
    }

    pub fn get_specialized_pipeline(
        &self,
        pipeline: &Handle<ComputePipelineDescriptor>,
        specialization: &ComputePipelineSpecialization,
    ) -> Option<Handle<ComputePipelineDescriptor>> {
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
        pipelines: &mut Assets<ComputePipelineDescriptor>,
        shaders: &mut Assets<Shader>,
        source_pipeline: &Handle<ComputePipelineDescriptor>,
        pipeline_specialization: &ComputePipelineSpecialization,
    ) -> Handle<ComputePipelineDescriptor> {
        let source_descriptor = pipelines.get(source_pipeline).unwrap();
        let mut specialized_descriptor = source_descriptor.clone();
        specialized_descriptor.shader_stages.compute = self.compile_shader(
            shaders,
            &specialized_descriptor.shader_stages.compute,
            &pipeline_specialization.shader_specialization,
        );

        //specialized_descriptor.reflect_layout(shaders, &pipeline_specialization.dynamic_bindings);

        let mut shader_layouts = vec![shaders
            .get(&specialized_descriptor.shader_stages.compute)
            .unwrap()
            .reflect_layout(true)
            .unwrap()];
        let mut layout = PipelineLayout::from_shader_layouts(&mut shader_layouts);

        // FIX: cut and paste
        if !pipeline_specialization.dynamic_bindings.is_empty() {
            // set binding uniforms to dynamic if render resource bindings use dynamic
            for bind_group in layout.bind_groups.iter_mut() {
                let mut binding_changed = false;
                for binding in bind_group.bindings.iter_mut() {
                    if pipeline_specialization
                        .dynamic_bindings
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
        specialized_descriptor.layout = Some(layout);

        let specialized_pipeline_handle = pipelines.add(specialized_descriptor);
        render_resource_context.create_compute_pipeline(
            specialized_pipeline_handle.clone_weak(),
            pipelines.get(&specialized_pipeline_handle).unwrap(),
            &shaders,
        );

        let specialized_pipelines = self
            .specialized_pipelines
            .entry(source_pipeline.clone_weak())
            .or_insert_with(Vec::new);
        let weak_specialized_pipeline_handle = specialized_pipeline_handle.clone_weak();
        specialized_pipelines.push(ComputeSpecializedPipeline {
            pipeline: specialized_pipeline_handle,
            specialization: pipeline_specialization.clone(),
        });

        weak_specialized_pipeline_handle
    }

    pub fn iter_compiled_pipelines(
        &self,
        pipeline_handle: Handle<ComputePipelineDescriptor>,
    ) -> Option<impl Iterator<Item = &Handle<ComputePipelineDescriptor>>> {
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

    pub fn iter_all_compiled_pipelines(
        &self,
    ) -> impl Iterator<Item = &Handle<ComputePipelineDescriptor>> {
        self.specialized_pipelines
            .values()
            .map(|compiled_pipelines| {
                compiled_pipelines
                    .iter()
                    .map(|specialized_pipeline| &specialized_pipeline.pipeline)
            })
            .flatten()
    }
}
