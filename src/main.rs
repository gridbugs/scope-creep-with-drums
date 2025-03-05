use bevy::{
    input::mouse::MouseMotion,
    prelude::*,
    window::{CursorGrabMode, WindowResolution},
};
use caw::prelude::*;
use core::f32;
use geom::{Circle, *};
use grid_2d::Coord;
use lazy_static::lazy_static;
use procgen::{Map1, Map2};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::{
    cmp::Ordering,
    collections::{HashSet, VecDeque},
    mem,
};

mod geom;
mod procgen;
mod text;

const DISPLAY_WIDTH: f32 = 960.;
const DISPLAY_HEIGHT: f32 = 720.;
const TOP_LEFT_OFFSET: Vec2 = Vec2::new(-DISPLAY_WIDTH / 2., DISPLAY_HEIGHT / 2.);
const MAX_NUM_SAMPLES: usize = 6_000;
const SCALE: f32 = 20.;

#[derive(PartialEq, Eq, Hash, Clone, Copy, Debug)]
struct HashableSeg {
    start: (i32, i32),
    end: (i32, i32),
}

impl HashableSeg {
    fn from_seg(seg: Seg2) -> Self {
        Self {
            start: (seg.start.x as i32, seg.start.y as i32),
            end: (seg.end.x as i32, seg.end.y as i32),
        }
    }
    fn to_seg(self) -> Seg2 {
        Seg2 {
            start: Vec2::new(self.start.0 as f32, self.start.1 as f32),
            end: Vec2::new(self.end.0 as f32, self.end.1 as f32),
        }
    }
}

#[derive(Clone)]
struct SceneTracer {
    scene: FrameSig<FrameSigVar<RenderedScene>>,
    buf: Vec<Vec2>,
    index: usize,
    text_index: usize,
}

const SCREEN_RIGHT: Seg2 = Seg2 {
    start: Vec2::new(DISPLAY_WIDTH / 2., DISPLAY_HEIGHT / 2.),
    end: Vec2::new(DISPLAY_WIDTH / 2., -DISPLAY_HEIGHT / 2.),
};

const SCREEN_BOTTOM: Seg2 = Seg2 {
    start: Vec2::new(DISPLAY_WIDTH / 2., -DISPLAY_HEIGHT / 2.),
    end: Vec2::new(-DISPLAY_WIDTH / 2., -DISPLAY_HEIGHT / 2.),
};
const SCREEN_LEFT: Seg2 = Seg2 {
    start: Vec2::new(-DISPLAY_WIDTH / 2., -DISPLAY_HEIGHT / 2.),
    end: Vec2::new(-DISPLAY_WIDTH / 2., DISPLAY_HEIGHT / 2.),
};
const SCREEN_TOP: Seg2 = Seg2 {
    start: Vec2::new(-DISPLAY_WIDTH / 2., DISPLAY_HEIGHT / 2.),
    end: Vec2::new(DISPLAY_WIDTH / 2., DISPLAY_HEIGHT / 2.),
};

fn clip_seg_within_display(mut s: Seg2) -> Seg2 {
    let pad = 20.;
    if let Some(clip) = s.intersect(&(SCREEN_LEFT.add_vec(Vec2::new(-pad, 0.)))) {
        *s.with_x_min() = clip;
    }
    if let Some(clip) = s.intersect(&(SCREEN_RIGHT.add_vec(Vec2::new(pad, 0.)))) {
        *s.with_x_max() = clip;
    }
    if let Some(clip) = s.intersect(&(SCREEN_BOTTOM.add_vec(Vec2::new(0., -pad)))) {
        *s.with_y_min() = clip;
    }
    if let Some(clip) = s.intersect(&(SCREEN_TOP.add_vec(Vec2::new(0., pad)))) {
        *s.with_y_max() = clip;
    }
    s
}

impl SigT for SceneTracer {
    type Item = Vec2;

    fn sample(&mut self, ctx: &SigCtx) -> impl Buf<Self::Item> {
        self.buf.clear();
        let scene: RenderedScene = self.scene.frame_sample(ctx);
        if scene.world.is_empty() {
            self.buf.resize(ctx.num_samples, Vec2::ZERO);
        } else {
            while self.buf.len() < (4 * ctx.num_samples) / 5 {
                let mut start = true;
                let rendered_world_seg = scene.world[self.index % scene.world.len()];
                let brightness = 20. / rendered_world_seg.mid_depth;
                let num_reps = (brightness as usize).clamp(1, 10) * 2;
                for _ in 0..num_reps {
                    let seg = clip_seg_within_display(rendered_world_seg.projected_seg);
                    if start {
                        self.buf.push(seg.start);
                    } else {
                        self.buf.push(seg.end);
                    }
                    start = !start;
                }
                self.index += 1;
            }
            while self.buf.len() < ctx.num_samples {
                let text_point = scene.text[self.text_index % scene.text.len()];
                self.buf.push(text_point);
                self.text_index += 1;
            }
        }
        &self.buf
    }
}

lazy_static! {
    static ref ARTIFACT_1_LABEL: Vec<Vec2> = render_text("ARTIFACT OF ORDER", Vec2::ZERO, 2);
    static ref ARTIFACT_1_LABEL_WIDTH: f32 = {
        ARTIFACT_1_LABEL
            .iter()
            .map(|v| v.x)
            .max_by(|a, b| a.total_cmp(b))
            .unwrap()
    };
    static ref ARTIFACT_2_LABEL: Vec<Vec2> = render_text("ARTIFACT OF HARMONY", Vec2::ZERO, 2);
    static ref ARTIFACT_2_LABEL_WIDTH: f32 = {
        ARTIFACT_2_LABEL
            .iter()
            .map(|v| v.x)
            .max_by(|a, b| a.total_cmp(b))
            .unwrap()
    };
    static ref ARTIFACT_3_LABEL: Vec<Vec2> = render_text("ARTIFACT OF CHAOS", Vec2::ZERO, 2);
    static ref ARTIFACT_3_LABEL_WIDTH: f32 = {
        ARTIFACT_3_LABEL
            .iter()
            .map(|v| v.x)
            .max_by(|a, b| a.total_cmp(b))
            .unwrap()
    };
    static ref HEALTH_SYMBOL: Vec<Vec2> = render_text("h", Vec2::ZERO, 2);
    static ref MANA_SYMBOL: Vec<Vec2> = render_text("s", Vec2::ZERO, 2);
}

#[derive(Clone)]
struct ObjectRenderer<O: FrameSigT<Item = Option<RenderedObject>>> {
    object: O,
    buf: Vec<Vec2>,
    sample_index: u64,
    text_index: usize,
    rng: StdRng,
}

impl<O: FrameSigT<Item = Option<RenderedObject>>> ObjectRenderer<O> {
    fn sample_artifact3(&mut self, object: &RenderedObject, ctx: &SigCtx) {
        let offset = Vec2::new(object.mid, -0.5 * object.height);
        for _ in 0..(ctx.num_samples / 2) {
            let speed = 100.;
            let r = self.rng.random::<f32>() * 2.0 - 1.0;
            let dx = ((speed * 60. * self.sample_index as f32) / ctx.sample_rate_hz).cos() * r;
            let dy = ((speed * 60. * self.sample_index as f32) / ctx.sample_rate_hz).sin() * r;
            let delta = Vec2::new(dx, dy) * 0.5;
            self.sample_index += 1;
            let mut v = offset + delta * object.height;
            v.x = v.x.clamp(object.right, object.left);
            self.buf.push(v);
        }
        while self.buf.len() < ctx.num_samples {
            let mut v = (ARTIFACT_3_LABEL[self.text_index % ARTIFACT_3_LABEL.len()]
                - Vec2::new(*ARTIFACT_3_LABEL_WIDTH / 2., 0.))
                * object.height
                * 0.005
                + offset
                + Vec2::new(0., object.height * 0.7);
            v.x = v.x.clamp(object.right, object.left);
            self.text_index += 1;
            self.buf.push(v);
        }
    }

    fn sample_artifact2(&mut self, object: &RenderedObject, ctx: &SigCtx) {
        let offset = Vec2::new(object.mid, -0.5 * object.height);
        for _ in 0..(ctx.num_samples / 2) {
            let speed = 100.;
            let dx = ((speed * 60. * self.sample_index as f32) / ctx.sample_rate_hz).cos();
            let dy = ((speed * 90.01 * self.sample_index as f32) / ctx.sample_rate_hz).sin();
            let delta = Vec2::new(dx, dy) * 0.5;
            self.sample_index += 1;
            let mut v = offset + delta * object.height;
            v.x = v.x.clamp(object.right, object.left);
            self.buf.push(v);
        }
        while self.buf.len() < ctx.num_samples {
            let mut v = (ARTIFACT_2_LABEL[self.text_index % ARTIFACT_2_LABEL.len()]
                - Vec2::new(*ARTIFACT_2_LABEL_WIDTH / 2., 0.))
                * object.height
                * 0.005
                + offset
                + Vec2::new(0., object.height * 0.7);
            v.x = v.x.clamp(object.right, object.left);
            self.text_index += 1;
            self.buf.push(v);
        }
    }

    fn sample_artifact1(&mut self, object: &RenderedObject, ctx: &SigCtx) {
        let offset = Vec2::new(object.mid, -0.5 * object.height);
        for _ in 0..(ctx.num_samples / 2) {
            let speed = 100.;
            let effect = ((speed * 2. * self.sample_index as f32) / ctx.sample_rate_hz).sin();
            let dx = ((speed * 60. * self.sample_index as f32) / ctx.sample_rate_hz).cos() * effect;
            let dy = ((speed * 60. * self.sample_index as f32) / ctx.sample_rate_hz).sin() * effect;
            let delta = Vec2::new(dx, dy) * 0.5;
            self.sample_index += 1;
            let mut v = offset + delta * object.height;
            v.x = v.x.clamp(object.right, object.left);
            self.buf.push(v);
        }
        while self.buf.len() < ctx.num_samples {
            let mut v = (ARTIFACT_1_LABEL[self.text_index % ARTIFACT_1_LABEL.len()]
                - Vec2::new(*ARTIFACT_1_LABEL_WIDTH / 2., 0.))
                * object.height
                * 0.005
                + offset
                + Vec2::new(0., object.height * 0.7);
            v.x = v.x.clamp(object.right, object.left);
            self.text_index += 1;
            self.buf.push(v);
        }
    }

    fn sample_ghost(&mut self, object: &RenderedObject, ctx: &SigCtx) {
        let offset = Vec2::new(object.mid, object.height * -0.2);
        for i in 0..ctx.num_samples {
            let random_01 = self.rng.random::<f32>();
            let rect = move |l: f32, t: f32, w: f32, h: f32| {
                Vec2::new(l, -t) + Vec2::new(w, h) * random_01
            };
            let v = match (i / 4) % 2 {
                0 => rect(-0.5, 1., 1., 2.),
                1 => rect(-0.25, -1.25, 0.5, 0.5),
                _ => unreachable!(),
            };
            let mut v = v * object.height + offset;
            v.x = v.x.clamp(object.right, object.left);
            self.buf.push(v);
        }
    }

    fn sample_slug(&mut self, object: &RenderedObject, ctx: &SigCtx) {
        let offset = Vec2::new(object.mid, object.height * -1.);
        for i in 0..ctx.num_samples {
            let delta = match (i / 8) % 3 {
                1 => {
                    let random_x = self.rng.random::<f32>() * 0.9;
                    let random_y = self.rng.random::<f32>() * 0.5;
                    let speed = 100.;
                    let dx = ((speed * 60. * self.sample_index as f32) / ctx.sample_rate_hz).cos()
                        * random_x;
                    let mut dy = ((speed * 60. * self.sample_index as f32) / ctx.sample_rate_hz)
                        .sin()
                        * random_y;
                    dy = dy.abs();
                    Vec2::new(dx, dy)
                }
                0 => {
                    let random_x = self.rng.random::<f32>() * 2. - 1.;
                    let random_y = self.rng.random::<f32>() * 2. - 1.;
                    let speed = 100.;
                    let dx = ((speed * 60. * self.sample_index as f32) / ctx.sample_rate_hz).cos()
                        * 0.2
                        + random_x * 0.05;
                    let dy = ((speed * 60. * self.sample_index as f32) / ctx.sample_rate_hz).sin()
                        * 0.2
                        + random_y * 0.05;
                    Vec2::new(dx, dy) + Vec2::new(0.5, 0.8)
                }
                2 => {
                    let random_x = self.rng.random::<f32>() * 2. - 1.;
                    let random_y = self.rng.random::<f32>() * 2. - 1.;
                    let speed = 100.;
                    let dx = ((speed * 60. * self.sample_index as f32) / ctx.sample_rate_hz).cos()
                        * 0.2
                        + random_x * 0.05;
                    let dy = ((speed * 60. * self.sample_index as f32) / ctx.sample_rate_hz).sin()
                        * 0.2
                        + random_y * 0.05;
                    Vec2::new(dx, dy) + Vec2::new(-0.5, 0.8)
                }
                _ => unreachable!(),
            };
            self.sample_index += 1;
            let mut v = offset + delta * object.height;
            v.x = v.x.clamp(object.right, object.left);
            self.buf.push(v);
        }
    }

    fn sample_health(&mut self, object: &RenderedObject, ctx: &SigCtx) {
        let offset = Vec2::new(object.mid, -0.5 * object.height);
        while self.buf.len() < ctx.num_samples {
            let mut v = (HEALTH_SYMBOL[self.text_index % HEALTH_SYMBOL.len()]
                - Vec2::new(-0.5, 0.))
                * object.height
                * 0.05
                + offset;
            v.x = v.x.clamp(object.right, object.left);
            self.text_index += 1;
            self.buf.push(v);
        }
    }

    fn sample_mana(&mut self, object: &RenderedObject, ctx: &SigCtx) {
        let offset = Vec2::new(object.mid, -0.5 * object.height);
        while self.buf.len() < ctx.num_samples {
            let mut v = (MANA_SYMBOL[self.text_index % MANA_SYMBOL.len()] - Vec2::new(-0.5, 0.))
                * object.height
                * 0.05
                + offset;
            v.x = v.x.clamp(object.right, object.left);
            self.text_index += 1;
            self.buf.push(v);
        }
    }
}

impl<O: FrameSigT<Item = Option<RenderedObject>>> SigT for ObjectRenderer<O> {
    type Item = Vec2;

    fn sample(&mut self, ctx: &SigCtx) -> impl Buf<Self::Item> {
        self.buf.clear();
        if let Some(object) = self.object.frame_sample(ctx) {
            match object.typ {
                ObjectType::Artifact1 => self.sample_artifact1(&object, ctx),
                ObjectType::Artifact2 => self.sample_artifact2(&object, ctx),
                ObjectType::Artifact3 => self.sample_artifact3(&object, ctx),
                ObjectType::Ghost => self.sample_ghost(&object, ctx),
                ObjectType::Slug => self.sample_slug(&object, ctx),
                ObjectType::Health => self.sample_health(&object, ctx),
                ObjectType::Mana => self.sample_mana(&object, ctx),
            }
        } else {
            self.buf.resize(ctx.num_samples, Vec2::ZERO);
        }
        &self.buf
    }
}

fn sig(
    scene: FrameSig<FrameSigVar<RenderedScene>>,
    dist_to_nearest_ghost: FrameSig<FrameSigVar<f32>>,
    player_alive: FrameSig<FrameSigVar<bool>>,
    player_damage: FrameSig<FrameSigVar<bool>>,
) -> StereoPair<SigBoxed<f32>> {
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
            text_index: 0,
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
                    text_index: 0,
                    rng: StdRng::from_rng(&mut rand::rng()),
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
        let ghost_noise_level = dist_to_nearest_ghost.clone().map(|d| {
            let min = 8.;
            if d > min { 0. } else { 0.03 * (min - d) / min }
        });
        let death_amp_env = (adsr_linear_01(player_alive.clone())
            .attack_s(0.1)
            .release_s(match channel {
                Channel::Left => 2.0,
                Channel::Right => 1.5,
            })
            .build()
            + 0.0001)
            .map(|x| x.min(1.));
        let damage_env = adsr_linear_01(player_damage.clone()).release_s(1.0).build();
        let death_noise_env = adsr_linear_01(player_alive.clone().map(|b| !b))
            .attack_s(2.)
            .build();
        (((((world + objects) * post_scale)
            + (noise::brown() * ghost_noise_level)
            + (noise::brown() * damage_env * 0.05)
            + (noise::brown() * death_noise_env * 4.))
            .clamp_symetric(3.)
            / SCALE)
            * death_amp_env)
            .boxed()
    })
}

struct AudioState {
    player: PlayerOwned,
    rendered_scene: FrameSig<FrameSigVar<RenderedScene>>,
    dist_to_nearest_ghost: FrameSig<FrameSigVar<f32>>,
    player_alive: FrameSig<FrameSigVar<bool>>,
    player_damage: FrameSig<FrameSigVar<bool>>,
}

impl AudioState {
    fn new(
        rendered_scene: FrameSig<FrameSigVar<RenderedScene>>,
        dist_to_nearest_ghost: FrameSig<FrameSigVar<f32>>,
        player_alive: FrameSig<FrameSigVar<bool>>,
        player_damage: FrameSig<FrameSigVar<bool>>,
    ) -> Self {
        let player = Player::new()
            .unwrap()
            .into_owned_stereo(
                sig(
                    rendered_scene.clone(),
                    dist_to_nearest_ghost.clone(),
                    player_alive.clone(),
                    player_damage.clone(),
                ),
                ConfigOwned {
                    system_latency_s: 0.0167,
                    visualization_data_policy: Some(VisualizationDataPolicy::All),
                },
            )
            .unwrap();
        Self {
            player,
            rendered_scene,
            dist_to_nearest_ghost,
            player_alive,
            player_damage,
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
    let dist_to_nearest_ghost = frame_sig_var(f32::INFINITY);
    let player_alive = frame_sig_var(true);
    let player_damage = frame_sig_var(true);
    world.insert_non_send_resource(AudioState::new(
        rendered_scene,
        dist_to_nearest_ghost,
        player_alive,
        player_damage,
    ));
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
    audio_state
        .dist_to_nearest_ghost
        .0
        .set(state.distance_from_player_to_nearest_ghost());
    audio_state.player_alive.0.set(state.player.alive);
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
            Color::srgba(current_color.x, current_color.y, current_color.z, 0.2),
        );
        prev = sample;
    }
}

#[derive(Debug)]
struct Meter {
    current: i32,
    max: i32,
}

impl Meter {
    fn incr(&mut self) {
        self.current = (self.current + 1).min(self.max);
    }
    fn decr(&mut self) {
        self.current = (self.current - 1).max(0);
    }
    fn is_max(&self) -> bool {
        self.current == self.max
    }
    fn is_zero(&self) -> bool {
        self.current == 0
    }
}

#[derive(Debug)]
struct PlayerCharacter {
    position: Vec2,
    facing_rad: f32,
    alive: bool,
    health: Meter,
    mana: Meter,
    iframes: u64,
}

impl PlayerCharacter {
    // The unit of [by] is an angle such that 1. is a reasonable amount for a single button press.
    // Positive values rotate to the right (clockwise looking down).
    fn rotate(&mut self, by: f32) {
        self.facing_rad += by * 0.005;
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
    fn take_damage(&mut self) {
        if self.iframes == 0 {
            self.health.decr();
            if self.health.is_zero() {
                self.alive = false;
            }
            self.iframes = 120;
        }
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
enum ObjectType {
    Artifact1,
    Artifact2,
    Artifact3,
    Ghost,
    Slug,
    Health,
    Mana,
}

#[derive(Clone, Copy, Debug)]
struct Object {
    typ: ObjectType,
    position: Vec2,
    radius: f32,
}

#[derive(Clone, Copy, Debug)]
struct ProjectedObject {
    typ: ObjectType,
    occluded: bool,
    screen_space_seg: Seg2,
    screen_space_position: Vec2,
}

#[derive(Clone, Copy, Debug)]
struct RenderedObject {
    typ: ObjectType,
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

fn render_text(text: &str, screen_coord: Vec2, num_reps: usize) -> Vec<Vec2> {
    let kerning = 0.2;
    let char_width = 0.8;
    let scale = 20.0;
    text.chars()
        .enumerate()
        .flat_map(|(i, ch)| {
            let shape = text::char_shape(ch);
            let shape = shape
                .iter()
                .cycle()
                .take(shape.len() * num_reps)
                .map(move |&v| {
                    let v = Vec2::new(v.x * char_width, -v.y);
                    screen_coord + (v + Vec2::new(i as f32 * (kerning + char_width), 0.)) * scale
                })
                .collect::<Vec<_>>();
            /*
            // Move the cursor below the letter so the bottom of the letter is visible boev the
            // connection between this and the following letter.
            if let Some(last) = shape.last() {
                shape.push(last - Vec2::new(0., SCALE * 0.3));
            }
            */
            shape
        })
        .collect()
}

#[derive(Clone, Default, Debug)]
struct RenderedScene {
    world: Vec<RenderedWorldSeg>,
    objects: Vec<RenderedObject>,
    text: Vec<Vec2>,
}

#[derive(Resource)]
struct State {
    map1: Map1,
    map2: Map2,
    player: PlayerCharacter,
    objects: Vec<Object>,
    paused: bool,
    show_map: bool,
    seen_walls: HashSet<HashableSeg>,
    frame_count: u64,
}

fn all_walls(map: &Map2) -> impl Iterator<Item = Seg2> {
    map.wall_strips
        .iter()
        .flat_map(|wall_strip| wall_strip.windows(2).map(|w| Seg2::new(w[0], w[1])))
}

fn all_walls_with_visible_corner(map: &Map2, player_position: Vec2) -> impl Iterator<Item = Seg2> {
    all_walls(map).filter(move |this| {
        let a = Seg2::new(player_position, this.start);
        let b = Seg2::new(player_position, this.end);
        for other in all_walls(map) {
            if other != *this {
                if let Some(a_ict) = other.intersect(&a) {
                    if this.start.distance(a_ict) > 0.01 {
                        return false;
                    }
                }
                if let Some(b_ict) = other.intersect(&b) {
                    if this.end.distance(b_ict) > 0.01 {
                        return false;
                    }
                }
            }
        }
        true
    })
}

const OBJECT_RADIUS: f32 = 0.25;

fn move_object_with_wall_collision_detection(mut position: Vec2, delta: Vec2, map: &Map2) -> Vec2 {
    let num_steps = 10;
    let mut step = delta / num_steps as f32;
    'outer: for _ in 0..num_steps {
        let test_position = position + step;
        for w in all_walls(map) {
            if w.grow(-0.01).overlaps_with_circle(&Circle {
                centre: test_position,
                radius: OBJECT_RADIUS,
            }) {
                step = step.project_onto(w.delta());
                continue 'outer;
            }
        }
        position = test_position;
    }
    position
}

impl State {
    fn new() -> Self {
        let map1 = Map1::new();
        let map2 = Map2::new();
        let player = PlayerCharacter {
            position: Vec2::new(12., 12.),
            facing_rad: 180f32.to_radians(),
            alive: true,
            health: Meter { current: 2, max: 2 },
            mana: Meter { current: 2, max: 2 },
            iframes: 0,
        };
        Self {
            map1,
            map2,
            player,
            objects: Vec::new(),
            paused: false,
            show_map: false,
            seen_walls: HashSet::new(),
            frame_count: 0,
        }
    }

    fn player_walk(&mut self, by: Vec2) {
        // Use right90 here so that (0, 1) represents forward, (1, 0) represents right, etc.
        let delta = self.player.right90().rotate(by) * 0.05;
        let num_steps = 10;
        let mut position = self.player.position;
        let mut step = delta / num_steps as f32;
        'outer: for _ in 0..num_steps {
            let test_position = position + step;
            for w in self.all_walls() {
                if w.grow(-0.01).overlaps_with_circle(&Circle {
                    centre: test_position,
                    radius: OBJECT_RADIUS,
                }) {
                    step = step.project_onto(w.delta());
                    continue 'outer;
                }
            }
            position = test_position;
        }
        self.player.position = position;
    }

    fn enemies_walk(&mut self) {
        for o in &mut self.objects {
            match o.typ {
                ObjectType::Ghost => {
                    if let Some(walk_delta) = (self.player.position - o.position).try_normalize() {
                        let walk_delta = walk_delta * 0.01;
                        o.position += walk_delta;
                    }
                }
                ObjectType::Slug => {
                    if let Some(walk_delta) = (self.player.position - o.position).try_normalize() {
                        let walk_delta = walk_delta * 0.02;
                        o.position = move_object_with_wall_collision_detection(
                            o.position, walk_delta, &self.map2,
                        );
                    }
                }
                ObjectType::Artifact1
                | ObjectType::Artifact2
                | ObjectType::Artifact3
                | ObjectType::Health
                | ObjectType::Mana => (),
            }
        }
    }

    fn distance_from_player_to_nearest_ghost(&self) -> f32 {
        let mut min_dist = f32::INFINITY;
        for o in &self.objects {
            if let ObjectType::Ghost = o.typ {
                let dist = o.position.distance(self.player.position);
                if dist < min_dist {
                    min_dist = dist;
                }
            }
        }
        min_dist
    }

    fn reset(&mut self) {
        let mut seed_rng = rand::rng();
        let seed = seed_rng.random();
        log::info!("seed: {:?}", seed);
        let mut rng = StdRng::from_seed(seed);
        let (map1, player_coord) = Map1::make_full(&mut rng);
        self.player.position = coord_to_vec(player_coord);
        self.player.facing_rad = 90f32.to_radians();
        self.map1 = map1;
        self.map2 = self.map1.to_map2();
        self.objects = vec![
            Object {
                typ: ObjectType::Health,
                position: self.player.position + Vec2::new(2., 3.),
                radius: 0.5,
            },
            Object {
                typ: ObjectType::Mana,
                position: self.player.position + Vec2::new(1., 3.),
                radius: 0.5,
            },
            Object {
                typ: ObjectType::Slug,
                position: self.player.position + Vec2::new(-2., 3.),
                radius: 0.5,
            },
            /*
            Object {
                typ: ObjectType::Artifact3,
                position: self.player.position + Vec2::new(0., 3.),
                radius: 0.5,
            },
            Object {
                typ: ObjectType::Ghost,
                position: self.player.position + Vec2::new(-10., 3.),
                radius: 0.5,
            }, */
        ];
        log::info!("Generated map!");
    }

    fn all_walls(&self) -> impl Iterator<Item = Seg2> {
        all_walls(&self.map2)
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
            if runs.len() >= 2 && &runs[0][0] == runs.last().unwrap().last().unwrap() {
                // The end of the last wall strip is the same as the start of the first wall strip.
                // This happens when a wall forming a closed loop is clipped, and now the region of
                // wall from the original starting point to the clipping plane, and the region of
                // wall from the clipping plane back to the original starting point are split into
                // two wall strips. Rejoin them in this case.
                let original_start = mem::take(&mut runs[0]);
                let mut original_end = runs.pop().unwrap();
                assert_eq!(original_start[0], original_end.pop().unwrap());
                original_end.extend(original_start);
                runs[0] = original_end;
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

    fn render_3d(&self) -> RenderedScene {
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
                    .any(|wall| ray_from_eye.intersect(&wall.grow(eps)).is_some())
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
                    typ: o.typ,
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
        let last_vertical_line = vertical_lines.last();
        let fixup = last_vertical_line
            .cloned()
            .zip(top_lines.last().cloned())
            .zip(bottom_lines.first().cloned())
            .map(|((mut vertical, mut top), mut bottom)| {
                if vertical.projected_seg.start.y < vertical.projected_seg.end.y {
                    vertical.projected_seg = vertical.projected_seg.flip();
                }
                top.projected_seg = top.projected_seg.flip();
                bottom.projected_seg = bottom.projected_seg.flip();
                vec![top, vertical, bottom]
            });
        world.extend(top_lines);
        if let Some(fixup) = fixup {
            world.extend(fixup);
        }
        world.extend(bottom_lines);
        world.extend(vertical_lines);
        let mut world2 = world.clone();
        world2.reverse();
        for w in &mut world2 {
            mem::swap(&mut w.projected_seg.start, &mut w.projected_seg.end);
        }
        world.extend(world2);
        let mut rendered_objects = Vec::new();
        for object in objects {
            let left = screen_space_project(object.screen_space_seg.start);
            let right = screen_space_project(object.screen_space_seg.end);
            let mid = screen_space_project(object.screen_space_position);
            if left.x > -DISPLAY_WIDTH / 2. && right.x < DISPLAY_WIDTH / 2. {
                rendered_objects.push(RenderedObject {
                    typ: object.typ,
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
            text: self.hud(),
        }
    }

    fn hud(&self) -> Vec<Vec2> {
        render_text(
            format!(
                "h{}/{} s{}/{}",
                self.player.health.current,
                self.player.health.max,
                self.player.mana.current,
                self.player.mana.max
            )
            .as_str(),
            Vec2::new(-DISPLAY_WIDTH / 2. + 10., -DISPLAY_HEIGHT / 2. + 20.),
            4,
        )
    }

    fn render_map(&self) -> RenderedScene {
        let player = vec![
            Seg2 {
                start: self.player.position - self.player.facing_vec2_normalized()
                    + self.player.right90() / 2.,
                end: self.player.position,
            },
            Seg2 {
                start: self.player.position
                    - self.player.facing_vec2_normalized()
                    - self.player.right90() / 2.,
                end: self.player.position,
            },
        ];
        let mut world = self
            .seen_walls
            .iter()
            .map(|s| s.to_seg())
            .chain(player)
            .flat_map(|s| {
                let seg = s.map(|v| (v - self.player.position) * 10.);
                let s = RenderedWorldSeg {
                    projected_seg: seg,
                    mid_depth: 0.,
                };
                (0..1).map(move |_| s)
            })
            .collect::<Vec<_>>();
        world.sort_by(|a, b| a.projected_seg.start.y.total_cmp(&b.projected_seg.start.y));
        RenderedScene {
            world,
            objects: Vec::new(),
            text: self.hud(),
        }
    }

    fn render(&self) -> RenderedScene {
        if self.show_map {
            self.render_map()
        } else {
            self.render_3d()
        }
    }

    fn passive_update(&mut self, audio_state: &mut AudioState) {
        let mut damage_this_frame = false;
        let mut to_remove = Vec::new();
        for (i, o) in self.objects.iter().enumerate() {
            match o.typ {
                ObjectType::Ghost | ObjectType::Slug => {
                    if o.position.distance(self.player.position) < OBJECT_RADIUS * 2. {
                        self.player.take_damage();
                        damage_this_frame = true;
                    }
                }
                ObjectType::Health => {
                    if o.position.distance(self.player.position) < OBJECT_RADIUS * 2.
                        && !self.player.health.is_max()
                    {
                        self.player.health.incr();
                        to_remove.push(i);
                    }
                }
                ObjectType::Mana => {
                    if o.position.distance(self.player.position) < OBJECT_RADIUS * 2.
                        && !self.player.mana.is_max()
                    {
                        self.player.mana.incr();
                        to_remove.push(i);
                    }
                }
                ObjectType::Artifact1 | ObjectType::Artifact2 | ObjectType::Artifact3 => (),
            }
        }
        for i in to_remove {
            self.objects.swap_remove(i);
        }
        audio_state.player_damage.0.set(damage_this_frame);
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
    let offset = TOP_LEFT_OFFSET + Vec2::new(40., -340.);
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
    let offset = TOP_LEFT_OFFSET + Vec2::new(650., -240.);
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
    let offset = TOP_LEFT_OFFSET + Vec2::new(200., -640.);
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
    let RenderedScene {
        world,
        objects,
        text: _,
    } = state.render();
    for RenderedWorldSeg {
        projected_seg: Seg2 { start, end },
        ..
    } in world
    {
        gizmos.line_2d(start, end, Color::srgb(0., 0., 1.));
    }
    for RenderedObject {
        typ: _,
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
    if keys.pressed(KeyCode::KeyP) {
        state.paused = !state.paused;
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
    let mut delta = Vec2::ZERO;
    if keys.pressed(KeyCode::KeyW) || keys.pressed(KeyCode::ArrowUp) {
        delta += Vec2::new(0., 1.);
    }
    if keys.pressed(KeyCode::KeyS) || keys.pressed(KeyCode::ArrowDown) {
        delta += Vec2::new(0., -1.);
    }
    if keys.pressed(KeyCode::KeyA) || keys.pressed(KeyCode::ArrowLeft) {
        delta += Vec2::new(-1., 0.);
    }
    if keys.pressed(KeyCode::KeyD) || keys.pressed(KeyCode::ArrowRight) {
        delta += Vec2::new(1., 0.);
    }
    let sprint = keys.pressed(KeyCode::ShiftLeft);
    state.show_map = keys.pressed(KeyCode::Tab);
    if delta != Vec2::ZERO {
        delta = delta.normalize();
    }
    if sprint {
        delta *= 2.;
    }
    state.player_walk(delta);
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

fn enemy_update(mut state: ResMut<State>) {
    if !state.paused {
        state.enemies_walk();
    }
}

fn passives_update(mut state: ResMut<State>, mut audio_state: NonSendMut<AudioState>) {
    state.passive_update(&mut audio_state);
}

fn update_general(mut state: ResMut<State>) {
    for w in all_walls_with_visible_corner(&state.map2, state.player.position).collect::<Vec<_>>() {
        state.seen_walls.insert(HashableSeg::from_seg(w));
    }
    state.frame_count += 1;
    state.player.iframes = state.player.iframes.saturating_sub(1);
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
        .add_systems(
            FixedUpdate,
            (
                //debug_render_map1,
                //debug_render_map2,
                //debug_render_map2_pruned,
                //debug_render_map2_3d,
                debug_update,
                caw_tick,
                grab_mouse,
                input_update,
                enemy_update,
                passives_update,
                update_general,
                render_scope,
            ),
        )
        .run();
}
