use super::{
    state_descriptors::{
        BlendDescriptor, BlendFactor, BlendOperation, ColorStateDescriptor, ColorWrite,
        CompareFunction, CullMode, DepthStencilStateDescriptor, FrontFace, IndexFormat,
        PrimitiveTopology, RasterizationStateDescriptor, StencilStateFaceDescriptor,
    },
    ComputePipelineSpecialization, InputStepMode, PipelineLayout, PipelineSpecialization,
    RenderPipelineSpecialization, StencilStateDescriptor, VertexBufferDescriptor,
};
use crate::{
    renderer::{RenderResourceBindings, RenderResourceContext},
    shader::{Shader, ShaderStages},
    texture::TextureFormat,
};
use bevy_asset::{Asset, Assets, Handle};
use bevy_reflect::TypeUuid;
use bevy_reflect::{Reflect, ReflectComponent};

pub trait Descriptor: std::fmt::Debug + Clone + Asset {
    type Specialization: PipelineSpecialization;

    // fn name() -> Option<&'a str>;
    // fn layout() -> Option<&'a PipelineLayout>;
    fn get_shader_stages<'a>(&'a self) -> &'a ShaderStages;
    // fn get_layout(&self) -> Option<&PipelineLayout>;
    // fn get_layout_mut(&mut self) -> Option<&mut PipelineLayout>;

    fn specialize_from(
        &self,
        specialization: &Self::Specialization,
        shader_stages: ShaderStages,
        layout: PipelineLayout,
    ) -> Self;

    fn create_pipeline(
        &self,
        render_resource_context: &dyn RenderResourceContext,
        handle: Handle<Self>,
        shaders: &Assets<Shader>,
    );
}

pub type RenderPipeline = Pipeline<PipelineDescriptor>;
pub type ComputePipeline = Pipeline<ComputePipelineDescriptor>;

#[derive(Debug, Clone, Reflect)]
pub struct Pipeline<T: Descriptor> {
    pub pipeline: Handle<T>,
    pub specialization: T::Specialization,
    /// used to track if PipelineSpecialization::dynamic_bindings is in sync with RenderResourceBindings
    pub dynamic_bindings_generation: usize,
}

// FIX:? Handle<Asset<T>> implements Default (irrespective of T), but
// deriving seems to insist T be Default
impl<T: Descriptor> Default for Pipeline<T> {
    fn default() -> Self {
        Self {
            pipeline: Default::default(),
            specialization: Default::default(),
            dynamic_bindings_generation: Default::default(),
        }
    }
}

impl<T: Descriptor> Pipeline<T> {
    pub fn new(pipeline: Handle<T>) -> Self {
        Pipeline {
            specialization: Default::default(),
            pipeline,
            dynamic_bindings_generation: std::usize::MAX,
        }
    }

    pub fn specialized(pipeline: Handle<T>, specialization: T::Specialization) -> Self {
        Pipeline {
            pipeline,
            specialization,
            dynamic_bindings_generation: std::usize::MAX,
        }
    }
}

pub type RenderPipelines = Pipelines<PipelineDescriptor>;
pub type ComputePipelines = Pipelines<ComputePipelineDescriptor>;

#[derive(Debug, Clone, Reflect)]
#[reflect(Component)]
pub struct Pipelines<T: Descriptor> {
    pub pipelines: Vec<Pipeline<T>>,
    #[reflect(ignore)]
    pub bindings: RenderResourceBindings,
}

impl<T: Descriptor> Pipelines<T> {
    pub fn from_pipelines(pipelines: Vec<Pipeline<T>>) -> Self {
        Self {
            pipelines,
            ..Default::default()
        }
    }

    pub fn from_handles<'a, I: IntoIterator<Item = &'a Handle<T>>>(handles: I) -> Self {
        Pipelines {
            pipelines: handles
                .into_iter()
                .map(|pipeline| Pipeline::new(pipeline.clone_weak()))
                .collect::<Vec<_>>(),
            ..Default::default()
        }
    }
}

impl<T: Descriptor> Default for Pipelines<T> {
    fn default() -> Self {
        Self {
            bindings: Default::default(),
            pipelines: vec![Pipeline::<T>::default()],
        }
    }
}

// Rename RenderPipelineDescriptor?
#[derive(Clone, Debug, TypeUuid)]
#[uuid = "ebfc1d11-a2a4-44cb-8f12-c49cc631146c"]
pub struct PipelineDescriptor {
    pub name: Option<String>,
    pub layout: Option<PipelineLayout>,
    pub shader_stages: ShaderStages,
    pub rasterization_state: Option<RasterizationStateDescriptor>,

    /// The primitive topology used to interpret vertices.
    pub primitive_topology: PrimitiveTopology,

    /// The effect of draw calls on the color aspect of the output target.
    pub color_states: Vec<ColorStateDescriptor>,

    /// The effect of draw calls on the depth and stencil aspects of the output target, if any.
    pub depth_stencil_state: Option<DepthStencilStateDescriptor>,

    /// The format of any index buffers used with this pipeline.
    pub index_format: IndexFormat,

    /// The number of samples calculated per pixel (for MSAA).
    pub sample_count: u32,

    /// Bitmask that restricts the samples of a pixel modified by this pipeline.
    pub sample_mask: u32,

    /// When enabled, produces another sample mask per pixel based on the alpha output value, that
    /// is AND-ed with the sample_mask and the primitive coverage to restrict the set of samples
    /// affected by a primitive.
    /// The implicit mask produced for alpha of zero is guaranteed to be zero, and for alpha of one
    /// is guaranteed to be all 1-s.
    pub alpha_to_coverage_enabled: bool,
}

impl PipelineDescriptor {
    pub fn new(shader_stages: ShaderStages) -> Self {
        PipelineDescriptor {
            name: None,
            layout: None,
            color_states: Vec::new(),
            depth_stencil_state: None,
            shader_stages,
            rasterization_state: None,
            primitive_topology: PrimitiveTopology::TriangleList,
            index_format: IndexFormat::Uint32,
            sample_count: 1,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        }
    }

    pub fn default_config(shader_stages: ShaderStages) -> Self {
        PipelineDescriptor {
            name: None,
            primitive_topology: PrimitiveTopology::TriangleList,
            layout: None,
            index_format: IndexFormat::Uint32,
            sample_count: 1,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
            rasterization_state: Some(RasterizationStateDescriptor {
                front_face: FrontFace::Ccw,
                cull_mode: CullMode::Back,
                depth_bias: 0,
                depth_bias_slope_scale: 0.0,
                depth_bias_clamp: 0.0,
                clamp_depth: false,
            }),
            depth_stencil_state: Some(DepthStencilStateDescriptor {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: CompareFunction::Less,
                stencil: StencilStateDescriptor {
                    front: StencilStateFaceDescriptor::IGNORE,
                    back: StencilStateFaceDescriptor::IGNORE,
                    read_mask: 0,
                    write_mask: 0,
                },
            }),
            color_states: vec![ColorStateDescriptor {
                format: TextureFormat::default(),
                color_blend: BlendDescriptor {
                    src_factor: BlendFactor::SrcAlpha,
                    dst_factor: BlendFactor::OneMinusSrcAlpha,
                    operation: BlendOperation::Add,
                },
                alpha_blend: BlendDescriptor {
                    src_factor: BlendFactor::One,
                    dst_factor: BlendFactor::One,
                    operation: BlendOperation::Add,
                },
                write_mask: ColorWrite::ALL,
            }],
            shader_stages,
        }
    }

    pub fn get_layout(&self) -> Option<&PipelineLayout> {
        self.layout.as_ref()
    }

    pub fn get_layout_mut(&mut self) -> Option<&mut PipelineLayout> {
        self.layout.as_mut()
    }
}

impl Descriptor for PipelineDescriptor {
    type Specialization = RenderPipelineSpecialization;

    fn get_shader_stages<'a>(&'a self) -> &'a ShaderStages {
        &self.shader_stages
    }

    fn specialize_from(
        &self,
        specialization: &Self::Specialization,
        shader_stages: ShaderStages,
        mut layout: PipelineLayout,
    ) -> Self {
        // create a vertex layout that provides all attributes from either the specialized vertex buffers or a zero buffer
        let mut specialized_descriptor = self.clone();

        // the vertex buffer descriptor of the mesh
        let mesh_vertex_buffer_descriptor = &specialization.vertex_buffer_descriptor;

        // the vertex buffer descriptor that will be used for this pipeline
        let mut compiled_vertex_buffer_descriptor = VertexBufferDescriptor {
            step_mode: InputStepMode::Vertex,
            stride: mesh_vertex_buffer_descriptor.stride,
            ..Default::default()
        };

        for shader_vertex_attribute in layout.vertex_buffer_descriptors.iter() {
            let shader_vertex_attribute = shader_vertex_attribute
                .attributes
                .get(0)
                .expect("Reflected layout has no attributes.");

            if let Some(target_vertex_attribute) = mesh_vertex_buffer_descriptor
                .attributes
                .iter()
                .find(|x| x.name == shader_vertex_attribute.name)
            {
                // copy shader location from reflected layout
                let mut compiled_vertex_attribute = target_vertex_attribute.clone();
                compiled_vertex_attribute.shader_location = shader_vertex_attribute.shader_location;
                compiled_vertex_buffer_descriptor
                    .attributes
                    .push(compiled_vertex_attribute);
            } else {
                panic!(
                    "Attribute {} is required by shader, but not supplied by mesh. Either remove the attribute from the shader or supply the attribute ({}) to the mesh.",
                    shader_vertex_attribute.name,
                    shader_vertex_attribute.name,
                );
            }
        }

        //TODO: add other buffers (like instancing) here
        let mut vertex_buffer_descriptors = Vec::<VertexBufferDescriptor>::default();
        vertex_buffer_descriptors.push(compiled_vertex_buffer_descriptor);

        layout.vertex_buffer_descriptors = vertex_buffer_descriptors;
        specialized_descriptor.sample_count = specialization.sample_count;
        specialized_descriptor.primitive_topology = specialization.primitive_topology;
        specialized_descriptor.index_format = specialization.index_format;
        specialized_descriptor.layout = Some(layout);
        specialized_descriptor.shader_stages = shader_stages;

        specialized_descriptor
    }

    fn create_pipeline(
        &self,
        render_resource_context: &dyn RenderResourceContext,
        handle: Handle<Self>,
        shaders: &Assets<Shader>,
    ) {
        render_resource_context.create_render_pipeline(handle, &self, shaders)
    }
}

/// Compute pipeline descriptor
#[derive(Clone, Debug, TypeUuid)]
#[uuid = "70c987c4-04cf-4ea1-bdaf-ab22d9329389"]
pub struct ComputePipelineDescriptor {
    pub name: Option<String>,
    pub layout: Option<PipelineLayout>,
    pub shader_stages: ShaderStages,
}

impl ComputePipelineDescriptor {
    pub fn new(shader_stages: ShaderStages) -> Self {
        ComputePipelineDescriptor {
            name: None,
            layout: None,
            shader_stages,
        }
    }

    pub fn get_layout(&self) -> Option<&PipelineLayout> {
        self.layout.as_ref()
    }

    pub fn get_layout_mut(&mut self) -> Option<&mut PipelineLayout> {
        self.layout.as_mut()
    }
}

impl Descriptor for ComputePipelineDescriptor {
    type Specialization = ComputePipelineSpecialization;

    fn get_shader_stages<'a>(&'a self) -> &'a ShaderStages {
        &self.shader_stages
    }

    fn specialize_from(
        &self,
        _specialization: &Self::Specialization,
        shader_stages: ShaderStages,
        layout: PipelineLayout,
    ) -> Self {
        Self {
            name: self.name.clone(),
            shader_stages,
            layout: Some(layout),
        }
    }

    fn create_pipeline(
        &self,
        render_resource_context: &dyn RenderResourceContext,
        handle: Handle<Self>,
        shaders: &Assets<Shader>,
    ) {
        render_resource_context.create_compute_pipeline(handle, &self, shaders)
    }
}
