use bevy::{
    input::mouse::MouseMotion,
    prelude::*,
    window::{CursorGrabMode, WindowResolution},
};
use caw::prelude::*;
use geom::*;
use grid_2d::Coord;
use procgen::{Map1, Map2};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::{cmp::Ordering, collections::VecDeque, mem};

const DISPLAY_WIDTH: f32 = 960.;
const DISPLAY_HEIGHT: f32 = 720.;
const TOP_LEFT_OFFSET: Vec2 = Vec2::new(-DISPLAY_WIDTH / 2., DISPLAY_HEIGHT / 2.);
const MAX_NUM_SAMPLES: usize = 4_000;
const SCALE: f32 = 40.;

mod geom;
mod procgen;

#[derive(Clone)]
struct SceneTracer {
    scene: FrameSig<FrameSigVar<RenderedScene>>,
    buf: Vec<Vec2>,
    index: usize,
}

impl SigT for SceneTracer {
    type Item = Vec2;

    fn sample(&mut self, ctx: &SigCtx) -> impl Buf<Self::Item> {
        self.buf.clear();
        let scene: RenderedScene = self.scene.frame_sample(ctx);
        if scene.world.is_empty() {
            self.buf.resize(ctx.num_samples, Vec2::ZERO);
        } else {
            while self.buf.len() < ctx.num_samples {
                let mut start = true;
                let rendered_world_seg = scene.world[self.index % scene.world.len()];
                let brightness = 20. / rendered_world_seg.mid_depth;
                let num_reps = (brightness as usize).clamp(1, 10) * 2;
                for _ in 0..num_reps {
                    let seg = rendered_world_seg.projected_seg;
                    if start {
                        self.buf.push(seg.start);
                    } else {
                        self.buf.push(seg.end);
                    }
                    start = !start;
                }
                self.index += 1;
            }
        }
        &self.buf
    }
}

#[derive(Clone)]
struct ObjectRenderer<O: FrameSigT<Item = Option<RenderedObject>>> {
    object: O,
    buf: Vec<Vec2>,
    sample_index: u64,
}

impl<O: FrameSigT<Item = Option<RenderedObject>>> SigT for ObjectRenderer<O> {
    type Item = Vec2;

    fn sample(&mut self, ctx: &SigCtx) -> impl Buf<Self::Item> {
        self.buf.clear();
        if let Some(object) = self.object.frame_sample(ctx) {
            let offset = Vec2::new(object.mid, 0.);
            for _ in 0..ctx.num_samples {
                let speed = 100.;
                let dx = ((speed * 60. * self.sample_index as f32) / ctx.sample_rate_hz).cos();
                let dy = ((speed * 90.01 * self.sample_index as f32) / ctx.sample_rate_hz).sin();
                let delta = Vec2::new(dx, dy);
                self.sample_index += 1;
                let mut v = offset + delta * object.height;
                v.x = v.x.clamp(object.right, object.left);
                self.buf.push(v);
            }
        } else {
            self.buf.resize(ctx.num_samples, Vec2::ZERO);
        }
        &self.buf
    }
}

fn sig(scene: FrameSig<FrameSigVar<RenderedScene>>) -> StereoPair<SigBoxed<f32>> {
    let get_nth_object = {
        let scene = scene.clone();
        move |i: usize| scene.map(move |scene| scene.objects.get(i).cloned())
    };
    let _num_visible_objects = {
        let scene = scene.clone();
        scene.map(move |scene| scene.objects.len()).shared()
    };
    let nth_object_exists_mul = |i| {
        (get_nth_object.clone())(i)
            .map(|o| if o.is_some() { 1. } else { 0. })
            .shared()
    };
    let max_num_objects = 9;
    Stereo::new_fn_channel(|channel| {
        let scene_tracer = SceneTracer {
            scene: scene.clone(),
            buf: Vec::new(),
            index: 0,
        };
        let base_scale = 0.;
        let post_scale = 0.001;
        let base = oscillator(Sine, 30.)
            .reset_offset_01(channel.circle_phase_offset_01())
            .build()
            * base_scale;
        let object_renderers = (0..max_num_objects)
            .map(|i| {
                Sig(ObjectRenderer {
                    object: (get_nth_object.clone())(i),
                    buf: Vec::new(),
                    sample_index: 0,
                })
                .shared()
            })
            .collect::<Vec<_>>();

        let make_pulse = |i: usize| {
            let obj_pulse_width = 0.1;
            (oscillator(Pulse, 60.)
                .pulse_width_01(obj_pulse_width)
                .reset_offset_01(-obj_pulse_width * i as f32)
                .build()
                .signed_to_01()
                .inv_01()
                * nth_object_exists_mul(i))
            .shared()
        };
        let object_pulses = (0..max_num_objects).map(make_pulse).collect::<Vec<_>>();
        let object_pulse_sum = object_pulses.iter().cloned().sum::<Sig<_>>();
        let world_pulse = (Sig(1.) - object_pulse_sum).shared();
        let dim_of_channel = move |v: Vec2| match channel {
            Channel::Left => v.x,
            Channel::Right => v.y,
        };
        let world = base
            .zip(scene_tracer.clone())
            .map(move |(audio_sample, scene_sample)| dim_of_channel(scene_sample) + audio_sample);
        let world = world * world_pulse.clone();
        let objects = object_renderers
            .into_iter()
            .zip(object_pulses)
            .map(|(object_renderer, object_pulse)| {
                object_renderer.clone().map(dim_of_channel) * object_pulse.clone()
            })
            .sum::<Sig<_>>();
        ((world + objects) * post_scale / SCALE)
            .clamp_symetric(0.5)
            .boxed()
    })
}

struct AudioState {
    player: PlayerOwned,
    rendered_scene: FrameSig<FrameSigVar<RenderedScene>>,
}

impl AudioState {
    fn new(rendered_scene: FrameSig<FrameSigVar<RenderedScene>>) -> Self {
        let player = Player::new()
            .unwrap()
            .into_owned_stereo(
                sig(rendered_scene.clone()),
                ConfigOwned {
                    system_latency_s: 0.0167,
                    visualization_data_policy: Some(VisualizationDataPolicy::All),
                },
            )
            .unwrap();
        Self {
            player,
            rendered_scene,
        }
    }

    fn tick(&mut self, scope_state: &mut ScopeState) {
        self.player.with_visualization_data_and_clear(|data| {
            for chunks in data.chunks_exact(2) {
                let x = chunks[0];
                let y = chunks[1];
                scope_state.samples.push_back(Vec2::new(x, y));
            }
        });
        while scope_state.samples.len() > MAX_NUM_SAMPLES {
            scope_state.samples.pop_front();
        }
    }
}

#[derive(Resource)]
struct ScopeState {
    samples: VecDeque<Vec2>,
}

impl ScopeState {
    fn new() -> Self {
        Self {
            samples: VecDeque::new(),
        }
    }
}

fn setup_caw_player(world: &mut World) {
    let rendered_scene = frame_sig_var(RenderedScene::default());
    world.insert_non_send_resource(AudioState::new(rendered_scene));
    world.insert_resource(ScopeState::new());
}

fn setup(mut commands: Commands) {
    commands.spawn(Camera2d);
}

fn caw_tick(
    state: Res<State>,
    mut audio_state: NonSendMut<AudioState>,
    mut scope_state: ResMut<ScopeState>,
) {
    audio_state.rendered_scene.0.set(state.render());
    audio_state.tick(&mut scope_state);
}

fn render_scope(scope_state: Res<ScopeState>, window: Query<&Window>, mut gizmos: Gizmos) {
    let color = Vec3::new(0., 1., 0.);
    let mut current_color = Vec3::ZERO;
    let color_step = color / scope_state.samples.len() as f32;
    let scale = window.single().width() * SCALE;
    let mut samples_iter = scope_state.samples.iter().map(|sample| sample * scale);
    let mut prev = if let Some(first) = samples_iter.next() {
        first
    } else {
        return;
    };
    for sample in samples_iter {
        current_color += color_step;
        gizmos.line_2d(
            prev,
            sample,
            Color::srgba(current_color.x, current_color.y, current_color.z, 0.1),
        );
        prev = sample;
    }
}

#[derive(Debug)]
struct PlayerCharacter {
    position: Vec2,
    facing_rad: f32,
}

impl PlayerCharacter {
    // The unit of [by] is an angle such that 1. is a reasonable amount for a single button press.
    // Positive values rotate to the right (clockwise looking down).
    fn rotate(&mut self, by: f32) {
        self.facing_rad += by * 0.005;
    }
    fn walk(&mut self, by: Vec2) {
        // Use right90 here so that (0, 1) represents forward, (1, 0) represents right, etc.
        let delta = self.right90().rotate(by) * 0.05;
        self.position += delta;
    }
    fn facing_vec2_normalized(&self) -> Vec2 {
        let x = self.facing_rad.cos();
        let y = self.facing_rad.sin();
        Vec2 { x, y }
    }
    fn facing_vec2_normalized_rev(&self) -> Vec2 {
        let Vec2 { x, y } = self.facing_vec2_normalized();
        // cos is symetric and sin(-y) = -sin(y)
        Vec2 { x, y: -y }
    }
    fn left90_rev(&self) -> Vec2 {
        let Vec2 { x, y } = self.facing_vec2_normalized_rev();
        Vec2 { x: -y, y: x }
    }
    fn right90(&self) -> Vec2 {
        let Vec2 { x, y } = self.facing_vec2_normalized();
        Vec2 { x: y, y: -x }
    }
    fn transform_abs_vec2_to_rel(&self, v: Vec2) -> Vec2 {
        // rotate by a 90 degree rotated facing vector so that y+ is forward
        self.left90_rev().rotate(v - self.position)
    }
    fn is_point_in_front_of(&self, v: Vec2) -> bool {
        self.transform_abs_vec2_to_rel(v).y >= 0.
    }
    fn debug_linestrip(&self) -> Vec<Vec2> {
        vec![
            self.position,
            self.position + self.facing_vec2_normalized() * 5.,
        ]
    }
}

#[derive(Debug)]
struct ConnectedPoint {
    point: Vec2,
    neighbours: Vec<Vec2>,
}

enum ConnectedPointClassification {
    Stop,
    ContinueLeft,
    ContinueRight,
}

impl ConnectedPoint {
    fn vec_from_linestrip(linestrip: &[Vec2]) -> Vec<Self> {
        if linestrip.is_empty() {
            return Vec::new();
        }
        if linestrip.len() >= 3 && linestrip[0] == linestrip[linestrip.len() - 1] {
            linestrip
                .iter()
                .enumerate()
                .map(|(i, &point)| {
                    let neighbours = vec![
                        linestrip[if i == 0 { linestrip.len() - 2 } else { i - 1 }],
                        linestrip[if i == linestrip.len() - 1 { 1 } else { i + 1 }],
                    ];
                    Self { point, neighbours }
                })
                .collect()
        } else {
            linestrip
                .iter()
                .enumerate()
                .map(|(i, &point)| {
                    let mut neighbours = Vec::new();
                    if i > 0 {
                        neighbours.push(linestrip[i - 1]);
                    }
                    if i < linestrip.len() - 1 {
                        neighbours.push(linestrip[i + 1]);
                    }
                    Self { point, neighbours }
                })
                .collect()
        }
    }

    // Does a ray cast from the eye to `self` stop at `self` or continue past it. Equivalent to
    // testing whether all neighbouring points lie on the same side of the eye->self vector.
    // Operates in screen space where the eye is at the origin and is looking in the (0, 1)
    // direction.
    fn classify_screen_space(&self) -> ConnectedPointClassification {
        let eps = 0.0001;
        let this_ratio = self.point.x / self.point.y;
        if self
            .neighbours
            .iter()
            .all(|n| (n.x / n.y) <= this_ratio + eps)
        {
            ConnectedPointClassification::ContinueRight
        } else if self
            .neighbours
            .iter()
            .all(|n| (n.x / n.y) >= this_ratio - eps)
        {
            ConnectedPointClassification::ContinueLeft
        } else {
            ConnectedPointClassification::Stop
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Object {
    position: Vec2,
    radius: f32,
}

#[derive(Clone, Copy, Debug)]
struct ProjectedObject {
    occluded: bool,
    screen_space_seg: Seg2,
    screen_space_position: Vec2,
}

#[derive(Clone, Copy, Default, Debug)]
struct RenderedObject {
    left: f32,
    right: f32,
    mid: f32,
    height: f32,
}

#[derive(Clone, Copy, Default, Debug)]
struct RenderedWorldSeg {
    projected_seg: Seg2,
    mid_depth: f32,
}

#[derive(Clone, Default, Debug)]
struct RenderedScene {
    world: Vec<RenderedWorldSeg>,
    objects: Vec<RenderedObject>,
}

#[derive(Resource)]
struct State {
    map1: Map1,
    map2: Map2,
    player: PlayerCharacter,
    objects: Vec<Object>,
}

impl State {
    fn new() -> Self {
        let map1 = Map1::new();
        let map2 = Map2::new();
        let player = PlayerCharacter {
            position: Vec2::new(12., 12.),
            facing_rad: 180f32.to_radians(),
        };
        let objects = vec![
            Object {
                position: Vec2::new(5., 10.),
                radius: 1.,
            },
            Object {
                position: Vec2::new(12., 12.),
                radius: 1.,
            },
        ];
        Self {
            map1,
            map2,
            player,
            objects,
        }
    }

    fn reset(&mut self) {
        let mut seed_rng = StdRng::seed_from_u64(1);
        let seed = seed_rng.random();
        log::info!("seed: {:?}", seed);
        let mut rng = StdRng::from_seed(seed);
        self.map1.generate(&mut rng);
        self.map2 = self.map1.to_map2();
    }

    fn prune_geometry(&self) -> Vec<Vec<Vec2>> {
        let mut pruned_walls = Vec::new();
        let clipping_plane_offset = Vec2::new(0., 0.1);
        for wall_strip in &self.map2.wall_strips {
            let wall_strip_rel = wall_strip
                .iter()
                .map(|v| self.player.transform_abs_vec2_to_rel(v) - clipping_plane_offset)
                .collect::<Vec<_>>();
            let _ = wall_strip; // prevent us from accidentally referring to the wrong wall strip later
            let mut runs = vec![Vec::new()];
            for w in wall_strip_rel.windows(2) {
                let s = Seg2::new(w[0], w[1]);
                if s.start.y >= 0. && s.end.y >= 0. {
                    // s is entirely in front of the eye
                    runs.last_mut().unwrap().push(s.start);
                } else if let Some(x_crossing) = s.crosses_x_axis_at() {
                    if s.start.y > s.end.y {
                        // went from in front of the eye to behind the eye
                        let last = runs.last_mut().unwrap();
                        last.push(s.start);
                        last.push(x_crossing);
                        runs.push(Vec::new());
                    } else {
                        // went from behind the eye to in front of the eye
                        runs.last_mut().unwrap().push(x_crossing);
                    }
                } else {
                    // entirely behind the eye
                }
            }
            if let Some(last) = wall_strip_rel.last() {
                if last.y >= 0. {
                    runs.last_mut().unwrap().push(*last);
                }
            }
            if runs.last().unwrap().is_empty() {
                runs.pop();
            }
            pruned_walls.extend(runs);
        }
        for linestrip in &mut pruned_walls {
            for v in linestrip {
                *v += clipping_plane_offset;
            }
        }
        pruned_walls
    }

    fn render(&self) -> RenderedScene {
        let eps = 0.0001;
        let pruned_walls = self.prune_geometry();
        let all_walls = pruned_walls
            .iter()
            .flat_map(|linestrip| seg2s_from_linestrip(linestrip))
            .collect::<Vec<_>>();
        let connected_points = pruned_walls
            .iter()
            .flat_map(|linestrip| ConnectedPoint::vec_from_linestrip(linestrip))
            .collect::<Vec<_>>();
        let visible_connected_points = connected_points
            .iter()
            .filter(|cp| {
                let ray_from_eye = Seg2::new(Vec2::ZERO, cp.point);
                !all_walls
                    .iter()
                    .filter(|wall| {
                        // Prevent the ray from being tested for intersections with the walls that its vertex
                        // is part of.
                        !(wall.start == cp.point || wall.end == cp.point)
                    })
                    .any(|wall| {
                        if ray_from_eye.intersect(&wall.grow(eps)).is_some() {
                            //log::info!("excluding {:?}", cp.point);
                            true
                        } else {
                            false
                        }
                    })
            })
            .collect::<Vec<_>>();
        let project_to_wall = |v: Vec2| {
            let v_norm = v.normalize();
            let max_depth_test_dist = 1000.;
            let distant_point = v_norm * max_depth_test_dist;
            let ray_from_eye = Seg2::new(Vec2::ZERO, distant_point);
            all_walls
                .iter()
                .filter(|wall| {
                    // Prevent the ray from being tested for intersections with the walls that its vertex
                    // is part of.
                    !(wall.start == v || wall.end == v)
                })
                .filter_map(|wall| ray_from_eye.intersect(&wall.grow(eps)))
                .min_by(|a, b| {
                    if a.length() < b.length() {
                        Ordering::Less
                    } else if a.length() > b.length() {
                        Ordering::Greater
                    } else {
                        Ordering::Equal
                    }
                })
        };
        let mut objects = self
            .objects
            .iter()
            .filter_map(|o| {
                let to_edge = self.player.right90() * o.radius;
                let screen_space_position = self.player.transform_abs_vec2_to_rel(o.position);
                let screen_space_seg = Seg2 {
                    start: self.player.transform_abs_vec2_to_rel(o.position + to_edge),
                    end: self.player.transform_abs_vec2_to_rel(o.position - to_edge),
                };
                if screen_space_seg.start.y < 0. || screen_space_seg.end.y < 0. {
                    return None;
                }
                Some(ProjectedObject {
                    screen_space_seg,
                    occluded: false,
                    screen_space_position,
                })
            })
            .collect::<Vec<_>>();
        let mut projected_points = visible_connected_points
            .iter()
            .map(|cp| {
                let ceiling = match cp.classify_screen_space() {
                    ConnectedPointClassification::Stop => VertexProjectionCeiling::BothTop,
                    ConnectedPointClassification::ContinueLeft => {
                        let wall_point = project_to_wall(cp.point);
                        project_through_objects(cp.point, wall_point, Side::Left, &mut objects);
                        match wall_point {
                            Some(p) => VertexProjectionCeiling::LeftWall(p),
                            None => VertexProjectionCeiling::LeftNone,
                        }
                    }
                    ConnectedPointClassification::ContinueRight => {
                        let wall_point = project_to_wall(cp.point);
                        project_through_objects(cp.point, wall_point, Side::Right, &mut objects);
                        match wall_point {
                            Some(p) => VertexProjectionCeiling::RightWall(p),
                            None => VertexProjectionCeiling::RightNone,
                        }
                    }
                };
                VertexProjection {
                    screen_space_coord: cp.point,
                    ceiling,
                }
            })
            .collect::<Vec<_>>();
        projected_points.sort_by(|a, b| {
            let ratio_a = a.screen_space_coord.x / a.screen_space_coord.y;
            let ratio_b = b.screen_space_coord.x / b.screen_space_coord.y;
            if ratio_a < ratio_b {
                Ordering::Less
            } else if ratio_a > ratio_b {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        });
        let objects = objects
            .into_iter()
            .filter(|o| {
                // The occluded flag is only set if there is a ray from the eye to some part of the
                // object.
                if o.occluded {
                    return true;
                }
                // At this point the object is either entirely visible or entirely occluded. Testing
                // the visibility of a single point on the object is sufficient to test whether the
                // entire object is visible.
                let v = Seg2::new(Vec2::ZERO, o.screen_space_seg.start);
                for wall in &all_walls {
                    if v.intersect(&wall.grow(eps)).is_some() {
                        return false;
                    }
                }
                true
            })
            .collect::<Vec<_>>();
        let scale_y = 200.;
        let scale_x = 400.;
        let screen_space_project = |v: Vec2| {
            let x = scale_x * v.x / v.y;
            let y = scale_y / v.y;
            Vec2 { x, y }
        };
        let mut vertical_lines = Vec::new();
        for projected_point in projected_points.iter() {
            let start = screen_space_project(projected_point.screen_space_coord);
            if start.x > -DISPLAY_WIDTH / 2. && start.x < DISPLAY_WIDTH / 2. {
                //log::info!("{:?}", start);
                let end = Vec2::new(start.x, -start.y);
                vertical_lines.push(RenderedWorldSeg {
                    projected_seg: Seg2::new(start, end),
                    mid_depth: projected_point.screen_space_coord.y,
                });
            }
        }
        let mut top_lines = Vec::new();
        let mut bottom_lines = Vec::new();
        for w in projected_points.windows(2) {
            let left_opt = w[0].right_screen_space_coord();
            let right_opt = w[1].left_screen_space_coord();
            if let Some((left, right)) = left_opt.zip(right_opt) {
                let mid_depth = (left.y + right.y) / 2.;
                let left = screen_space_project(left);
                let right = screen_space_project(right);
                if right.x > -DISPLAY_WIDTH / 2. && left.x < DISPLAY_WIDTH / 2. {
                    top_lines.push(RenderedWorldSeg {
                        projected_seg: Seg2::new(left, right),
                        mid_depth,
                    });
                    let left = Vec2::new(left.x, -left.y);
                    let right = Vec2::new(right.x, -right.y);
                    // the bottom lines are put in backwards to reduce noise
                    bottom_lines.push(RenderedWorldSeg {
                        projected_seg: Seg2::new(right, left),
                        mid_depth,
                    });
                }
            }
        }
        // flip every second vertical line so there's less noise during rendering
        for (i, seg) in vertical_lines.iter_mut().enumerate() {
            if i % 2 == 0 {
                mem::swap(&mut seg.projected_seg.start, &mut seg.projected_seg.end);
            }
        }
        // reverse the lines along the bottom of the image to reduce rendering noise
        bottom_lines.reverse();
        let mut world = Vec::new();
        world.extend(vertical_lines);
        world.extend(top_lines);
        world.extend(bottom_lines);
        let mut rendered_objects = Vec::new();
        for object in objects {
            let left = screen_space_project(object.screen_space_seg.start);
            let right = screen_space_project(object.screen_space_seg.end);
            let mid = screen_space_project(object.screen_space_position);
            if left.x > -DISPLAY_WIDTH / 2. && right.x < DISPLAY_WIDTH / 2. {
                rendered_objects.push(RenderedObject {
                    left: left.x,
                    right: right.x,
                    mid: mid.x,
                    height: left.y,
                });
            }
        }
        RenderedScene {
            world,
            objects: rendered_objects,
        }
    }
}

fn setup_state(world: &mut World) {
    let mut state = State::new();
    state.reset();
    world.insert_resource(state);
}

fn coord_to_vec(coord: Coord) -> Vec2 {
    Vec2::new(coord.x as f32, coord.y as f32)
}

#[allow(unused)]
fn debug_render_map1(state: Res<State>, mut gizmos: Gizmos) {
    let cell_size = Vec2::new(5., 5.);
    let offset = TOP_LEFT_OFFSET + Vec2::new(40., -140.);
    for (coord, &cell) in state.map1.grid.enumerate() {
        if cell {
            gizmos.rect_2d(
                coord_to_vec(coord) * cell_size + offset,
                cell_size,
                Color::srgb(1., 1., 0.),
            );
        }
    }
    gizmos.linestrip_2d(
        state
            .player
            .debug_linestrip()
            .into_iter()
            .map(|v| cell_size * v + offset),
        Color::srgb(1., 0., 0.),
    );
    for object in &state.objects {
        gizmos.circle_2d(
            object.position * cell_size.x + offset,
            object.radius,
            Color::srgb(0., 1., 1.),
        );
    }
}

#[allow(unused)]
fn debug_render_map2(state: Res<State>, mut gizmos: Gizmos) {
    let wall_length = 5.;
    let offset = TOP_LEFT_OFFSET + Vec2::new(390., -140.);
    let transform = |v| (state.player.transform_abs_vec2_to_rel(v)) * wall_length + offset;
    for wall_strip in &state.map2.wall_strips {
        gizmos.linestrip_2d(wall_strip.iter().map(transform), Color::srgb(0., 1., 1.));
        for &v in wall_strip {
            if state.player.is_point_in_front_of(v) {
                gizmos.circle_2d(transform(v), 2., Color::srgb(1., 0., 0.));
            }
        }
    }
    gizmos.linestrip_2d(
        state.player.debug_linestrip().into_iter().map(transform),
        Color::srgb(1., 0., 0.),
    );
}

#[allow(unused)]
fn debug_render_map2_pruned(state: Res<State>, mut gizmos: Gizmos) {
    let pruned_walls = state.prune_geometry();
    let wall_length = 5.;
    let offset = TOP_LEFT_OFFSET + Vec2::new(640., -140.);
    let transform = |v| v * wall_length + offset;
    for wall_strip in pruned_walls {
        gizmos.linestrip_2d(
            wall_strip.iter().cloned().map(transform),
            Color::srgb(1., 0., 1.),
        );
        for v in wall_strip {
            if v.y >= 0. {
                gizmos.circle_2d(transform(v), 1., Color::srgb(0., 1., 1.));
            }
        }
    }
    gizmos.linestrip_2d(
        vec![transform(Vec2::ZERO), transform(Vec2::new(0., 5.))],
        Color::srgb(1., 0., 0.),
    );
}

fn seg2s_from_linestrip(linestrip: &[Vec2]) -> Vec<Seg2> {
    linestrip
        .windows(2)
        .map(|w| Seg2::new(w[0], w[1]))
        .collect::<Vec<_>>()
}

#[derive(Debug, Clone, Copy)]
enum VertexProjectionCeiling {
    BothTop,
    LeftWall(Vec2),
    RightWall(Vec2),
    LeftNone,
    RightNone,
}

#[derive(Debug)]
struct VertexProjection {
    screen_space_coord: Vec2,
    ceiling: VertexProjectionCeiling,
}

impl VertexProjection {
    fn left_screen_space_coord(&self) -> Option<Vec2> {
        match self.ceiling {
            VertexProjectionCeiling::BothTop
            | VertexProjectionCeiling::RightWall(_)
            | VertexProjectionCeiling::RightNone => Some(self.screen_space_coord),
            VertexProjectionCeiling::LeftWall(left) => Some(left),
            VertexProjectionCeiling::LeftNone => None,
        }
    }
    fn right_screen_space_coord(&self) -> Option<Vec2> {
        match self.ceiling {
            VertexProjectionCeiling::BothTop
            | VertexProjectionCeiling::LeftWall(_)
            | VertexProjectionCeiling::LeftNone => Some(self.screen_space_coord),
            VertexProjectionCeiling::RightWall(right) => Some(right),
            VertexProjectionCeiling::RightNone => None,
        }
    }
}

enum Side {
    Left,
    Right,
}

fn project_through_objects(
    v: Vec2,
    wall_point: Option<Vec2>,
    side: Side,
    objects: &mut Vec<ProjectedObject>,
) {
    let eps = 0.0001;
    let v_norm = v.normalize();
    let max_depth_test_dist = 1000.;
    let distant_point = v_norm * max_depth_test_dist;
    let ray_from_eye = Seg2::new(Vec2::ZERO, distant_point);
    for o in objects {
        if let Some(ict) = ray_from_eye.intersect(&o.screen_space_seg.grow(eps)) {
            if let Some(wall_point) = wall_point {
                if wall_point.y < ict.y {
                    // If the nearest wall to the eye along this ray is closer than the
                    // intersection point with the object then don't consider the intersection
                    // point with the object.
                    continue;
                }
            }
            // only occlude objects behind the wall
            if ict.y >= v.y {
                o.occluded = true;
                match side {
                    Side::Left => o.screen_space_seg.start = ict,
                    Side::Right => o.screen_space_seg.end = ict,
                }
            }
        }
    }
}

#[allow(unused)]
fn debug_render_map2_3d(state: Res<State>, mut gizmos: Gizmos) {
    let RenderedScene { world, objects } = state.render();
    for RenderedWorldSeg {
        projected_seg: Seg2 { start, end },
        ..
    } in world
    {
        gizmos.line_2d(start, end, Color::srgb(0., 1., 0.));
    }
    for RenderedObject {
        left,
        right,
        mid,
        height,
    } in objects
    {
        gizmos.line_2d(
            Vec2::new(left, height),
            Vec2::new(left, -height),
            Color::srgb(0., 1., 1.),
        );
        gizmos.line_2d(
            Vec2::new(right, height),
            Vec2::new(right, -height),
            Color::srgb(0., 1., 1.),
        );
        gizmos.line_2d(
            Vec2::new(mid, height),
            Vec2::new(mid, -height),
            Color::srgb(0., 1., 1.),
        );
    }
}

#[allow(unused)]
fn debug_update(mut state: ResMut<State>, keys: Res<ButtonInput<KeyCode>>) {
    if keys.just_pressed(KeyCode::KeyR) {
        state.reset();
    }
    if keys.pressed(KeyCode::KeyQ) {
        state.player.rotate(1.);
    }
    if keys.pressed(KeyCode::KeyE) {
        state.player.rotate(-1.);
    }
}

fn input_update(
    mut state: ResMut<State>,
    mut evr_motion: EventReader<MouseMotion>,
    keys: Res<ButtonInput<KeyCode>>,
) {
    for ev in evr_motion.read() {
        state.player.rotate(-ev.delta.x);
    }
    if keys.pressed(KeyCode::KeyW) || keys.pressed(KeyCode::ArrowUp) {
        state.player.walk(Vec2::new(0., 1.));
    }
    if keys.pressed(KeyCode::KeyS) || keys.pressed(KeyCode::ArrowDown) {
        state.player.walk(Vec2::new(0., -1.));
    }
    if keys.pressed(KeyCode::KeyA) || keys.pressed(KeyCode::ArrowLeft) {
        state.player.walk(Vec2::new(-1., 0.));
    }
    if keys.pressed(KeyCode::KeyD) || keys.pressed(KeyCode::ArrowRight) {
        state.player.walk(Vec2::new(1., 0.));
    }
}

// This system grabs the mouse when the left mouse button is pressed
// and releases it when the escape key is pressed
fn grab_mouse(
    mut window: Single<&mut Window>,
    mouse: Res<ButtonInput<MouseButton>>,
    key: Res<ButtonInput<KeyCode>>,
) {
    if mouse.just_pressed(MouseButton::Left) {
        window.cursor_options.visible = false;
        window.cursor_options.grab_mode = CursorGrabMode::Locked;
    }
    if key.just_pressed(KeyCode::Escape) {
        window.cursor_options.visible = true;
        window.cursor_options.grab_mode = CursorGrabMode::None;
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Scope Creep".into(),
                resolution: WindowResolution::new(DISPLAY_WIDTH, DISPLAY_HEIGHT),
                resizable: false,
                ..default()
            }),
            ..default()
        }))
        .add_systems(Startup, (setup_caw_player, setup, setup_state))
        .insert_resource(ClearColor(Color::srgb(0., 0., 0.)))
        .add_systems(FixedFirst, caw_tick)
        .add_systems(
            Update,
            (
                //debug_render_map1,
                //debug_render_map2,
                //debug_render_map2_pruned,
                //debug_render_map2_3d,
                //debug_update,
                grab_mouse,
                input_update,
                //caw_tick,
                render_scope,
            ),
        )
        .run();
}
