use bevy::prelude::*;
use caw::prelude::*;
use caw_bevy::BevyInput;
use std::collections::VecDeque;

const MAX_NUM_SAMPLES: usize = 4_000;

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

#[derive(Clone, Copy, Debug)]
struct Seg2 {
    start: Vec2,
    end: Vec2,
}

impl Seg2 {
    fn new(start: Vec2, end: Vec2) -> Self {
        Self { start, end }
    }
    fn delta(&self) -> Vec2 {
        self.end - self.start
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

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Oscillographics Cube".into(),
                resolution: (960., 720.).into(),
                ..default()
            }),
            ..default()
        }))
        .add_systems(Startup, (setup_caw_player, setup))
        .insert_resource(ClearColor(Color::srgb(0., 0., 0.)))
        .add_systems(FixedFirst, caw_tick)
        .add_systems(Update, (BevyInput::update, render_scope))
        .run();
}
