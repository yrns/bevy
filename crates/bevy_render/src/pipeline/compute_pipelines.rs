use super::ComputePipelines;
use crate::{
    dispatch::{Dispatch, DispatchContext},
    renderer::RenderResourceBindings,
};
use bevy_ecs::{Query, ResMut};

pub fn dispatch_compute_pipelines_system(
    mut dispatch_context: DispatchContext,
    mut render_resource_bindings: ResMut<RenderResourceBindings>,
    mut query: Query<(&mut Dispatch, &mut ComputePipelines)>,
) {
    for (mut dispatch, mut render_pipelines) in query.iter_mut() {
        let render_pipelines = &mut *render_pipelines;

        for render_pipeline in render_pipelines.pipelines.iter() {
            dispatch_context
                .set_pipeline(
                    &mut dispatch,
                    &render_pipeline.pipeline,
                    &render_pipeline.specialization,
                )
                .unwrap();
            dispatch_context
                .set_bind_groups_from_bindings(
                    &mut dispatch,
                    &mut [
                        &mut render_pipelines.bindings,
                        &mut render_resource_bindings,
                    ],
                )
                .unwrap();
            dispatch.dispatch();
        }
    }
}
