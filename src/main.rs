use bevy::{prelude::*, window::WindowResolution};
use caw::prelude::*;
use caw_bevy::BevyInput;
use geom::*;
use grid_2d::Coord;
use procgen::{Map1, Map2};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::collections::VecDeque;

const MAX_NUM_SAMPLES: usize = 4_000;

mod geom;
mod procgen;

#[derive(Clone, Copy, Debug)]
struct Seg3 {
    start: Vec3,
    end: Vec3,
}

impl Seg3 {
    fn new(start: Vec3, end: Vec3) -> Self {
        Self { start, end }
    }
}

fn cube_edges() -> Vec<Seg3> {
    let square_corners_horizontal = |y| {
        vec![
            Vec3::new(1., y, 1.),
            Vec3::new(1., y, -1.),
            Vec3::new(-1., y, -1.),
            Vec3::new(-1., y, 1.),
        ]
    };
    let top = square_corners_horizontal(1.);
    let btm = square_corners_horizontal(-1.);
    vec![
        // top face
        Seg3::new(top[0], top[1]),
        Seg3::new(top[1], top[2]),
        Seg3::new(top[2], top[3]),
        Seg3::new(top[3], top[0]),
        // bottom face
        Seg3::new(btm[0], btm[1]),
        Seg3::new(btm[1], btm[2]),
        Seg3::new(btm[2], btm[3]),
        Seg3::new(btm[3], btm[0]),
        // connections from top to bottom
        Seg3::new(top[0], btm[0]),
        Seg3::new(top[1], btm[1]),
        Seg3::new(top[2], btm[2]),
        Seg3::new(top[3], btm[3]),
    ]
}

fn transform_vec3_to_render(in_: Vec3, rot_y: f32, rot_z: f32) -> Vec2 {
    let transform = Mat4::from_rotation_x(rot_z);
    let in_ = transform.transform_point3(in_);
    let transform =
        Mat4::from_rotation_translation(Quat::from_rotation_y(rot_y), Vec3::new(0., 0., 2.2));
    let in_ = transform.transform_point3(in_);
    //let in_ = in_ + Vec3::new(0., 0., 4.);
    let perspective = Mat4::perspective_lh(std::f32::consts::FRAC_PI_4, 1., 0., 1.);
    let out = perspective.project_point3(in_);
    Vec2::new(out.x, out.y)
}

struct Cube<SY: FrameSigT<Item = f32>, SZ: FrameSigT<Item = f32>> {
    geometry: Vec<Seg3>,
    buf: Vec<Seg2>,
    rot_y: f32,
    rot_z: f32,
    speed_y: SY,
    speed_z: SZ,
}

impl<SY: FrameSigT<Item = f32>, SZ: FrameSigT<Item = f32>> Cube<SY, SZ> {
    fn new(speed_y: SY, speed_z: SZ) -> Self {
        Self {
            geometry: cube_edges(),
            buf: Vec::new(),
            rot_y: 0.,
            rot_z: 0.,
            speed_y,
            speed_z,
        }
    }
}

impl<SY: FrameSigT<Item = f32>, SZ: FrameSigT<Item = f32>> SigT for Cube<SY, SZ> {
    type Item = Seg2;

    fn sample(&mut self, ctx: &SigCtx) -> impl Buf<Self::Item> {
        self.buf.clear();
        let num_segment_repeat = ctx.num_samples / self.geometry.len();
        for i in 0..ctx.num_samples {
            let edge3 = self.geometry[(i / num_segment_repeat) % self.geometry.len()];
            let edge2 = Seg2::new(
                transform_vec3_to_render(edge3.start, self.rot_y, self.rot_z),
                transform_vec3_to_render(edge3.end, self.rot_y, self.rot_z),
            );
            self.buf.push(edge2);
        }
        self.rot_y += 0.05; // (self.speed_y.frame_sample(ctx) - 0.5) * 0.5;
        self.rot_z += 0.01; //(self.speed_z.frame_sample(ctx) - 0.5) * 0.5;
        &self.buf
    }
}

fn apply_effects(
    sig: Sig<impl SigT<Item = f32>>,
    mouse_x: Sig<impl SigT<Item = f32>>,
    mouse_y: Sig<impl SigT<Item = f32>>,
) -> Sig<impl SigT<Item = f32>> {
    sig.filter(low_pass::default(20000. * mouse_y).resonance(mouse_x))
}

fn sig(
    BevyInput {
        mouse_x,
        mouse_y,
        keyboard,
    }: BevyInput,
) -> StereoPair<SigBoxed<f32>> {
    let shape = Sig(Cube::new(mouse_x.clone(), mouse_y.clone())).shared();
    Stereo::new_fn_channel(|channel| {
        let MonoVoice {
            note,
            key_down_gate,
            key_press_trig,
            ..
        } = keyboard
            .clone()
            .opinionated_key_events(Note::B0)
            .mono_voice();
        let note = note.shared();
        let env = adsr_linear_01(key_down_gate)
            .key_press_trig(key_press_trig)
            .attack_s(1.)
            .release_s(4.)
            .build()
            .exp_01(1.0)
            .shared();
        let sig = oscillator(Saw, note.clone().freq_hz() * 4.)
            .build()
            //.filter(chorus().num_voices(1).lfo_rate_hz(0.05).delay_s(0.01))
            //.filter(low_pass::default(5000.).resonance(0.5))
            .filter(compressor().threshold(1.).ratio(0.0).scale(2.))
            .signed_to_01();
        let scale = 0.1;
        match channel {
            Channel::Left => {
                let sig = sig.zip(shape.clone()).map(|(audio_sample, shape_sample)| {
                    let delta = shape_sample.delta();
                    (audio_sample * delta.x) + shape_sample.start.x
                }) * scale;
                let sig = apply_effects(
                    sig,
                    Sig(mouse_x.clone()),
                    env.clone() * Sig(mouse_y.clone()),
                );
                sig.boxed()
            }
            Channel::Right => {
                let sig = sig.zip(shape.clone()).map(|(audio_sample, shape_sample)| {
                    let delta = shape_sample.delta();
                    (audio_sample * delta.y) + shape_sample.start.y
                }) * scale;
                let sig = apply_effects(
                    sig,
                    Sig(mouse_x.clone()),
                    env.clone() * Sig(mouse_y.clone()),
                );
                sig.boxed()
            }
        }
    })
}

struct AudioState {
    player: PlayerOwned,
}

impl AudioState {
    fn new(input: BevyInput) -> Self {
        let player = Player::new()
            .unwrap()
            .into_owned_stereo(
                sig(input),
                ConfigOwned {
                    system_latency_s: 0.0167,
                    visualization_data_policy: Some(VisualizationDataPolicy::All),
                },
            )
            .unwrap();
        Self { player }
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
    let input = BevyInput::default();
    world.insert_non_send_resource(AudioState::new(input.clone()));
    world.insert_resource(input);
    world.insert_resource(ScopeState::new());
}

fn setup(mut commands: Commands) {
    commands.spawn(Camera2d);
}

fn caw_tick(mut audio_state: NonSendMut<AudioState>, mut scope_state: ResMut<ScopeState>) {
    audio_state.tick(&mut scope_state);
}

fn render_scope(scope_state: Res<ScopeState>, window: Query<&Window>, mut gizmos: Gizmos) {
    let color = Vec3::new(0., 1., 0.);
    let mut current_color = Vec3::ZERO;
    let color_step = color / scope_state.samples.len() as f32;
    let scale = window.single().width();
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

struct PlayerCharacter {
    position: Vec2,
    facing_rad: f32,
}

impl PlayerCharacter {
    // The unit of [by] is an angle such that 1. is a reasonable amount for a single button press.
    // Positive values rotate to the right (clockwise looking down).
    fn rotate(&mut self, by: f32) {
        self.facing_rad += by * 0.01;
    }
    fn walk(&mut self, by: Vec2) {
        // Use right90 here so that (0, 1) represents forward, (1, 0) represents right, etc.
        let delta = self.right90().rotate(by) * 0.1;
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

#[derive(Resource)]
struct State {
    map1: Map1,
    map2: Map2,
    player: PlayerCharacter,
}

impl State {
    fn new() -> Self {
        let map1 = Map1::new();
        let map2 = Map2::new();
        let player = PlayerCharacter {
            position: Vec2::new(15., 10.),
            facing_rad: 0f32.to_radians(),
        };
        Self { map1, map2, player }
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
        for wall_strip in &self.map2.wall_strips {
            let wall_strip_rel = wall_strip
                .iter()
                .map(|v| self.player.transform_abs_vec2_to_rel(v))
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
        pruned_walls
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

const DISPLAY_WIDTH: f32 = 960.;
const DISPLAY_HEIGHT: f32 = 720.;

const TOP_LEFT_OFFSET: Vec2 = Vec2::new(-DISPLAY_WIDTH / 2., DISPLAY_HEIGHT / 2.);

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
                gizmos.circle_2d(transform(v), 2., Color::srgb(0., 1., 1.));
            }
        }
    }
    gizmos.linestrip_2d(
        vec![transform(Vec2::ZERO), transform(Vec2::new(0., 5.))],
        Color::srgb(1., 0., 0.),
    );
}

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
    if keys.pressed(KeyCode::KeyW) {
        state.player.walk(Vec2::new(0., 1.));
    }
    if keys.pressed(KeyCode::KeyS) {
        state.player.walk(Vec2::new(0., -1.));
    }
    if keys.pressed(KeyCode::KeyA) {
        state.player.walk(Vec2::new(-1., 0.));
    }
    if keys.pressed(KeyCode::KeyD) {
        state.player.walk(Vec2::new(1., 0.));
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
        //.add_systems(FixedFirst, caw_tick)
        //.add_systems(Update, BevyInput::update)
        .add_systems(
            Update,
            (
                debug_render_map1,
                debug_render_map2,
                debug_render_map2_pruned,
                debug_update,
            ),
        )
        .run();
}
