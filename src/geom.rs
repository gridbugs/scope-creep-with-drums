use bevy::math::Vec2;

#[derive(Clone, Copy, Debug)]
pub struct Seg2 {
    pub start: Vec2,
    pub end: Vec2,
}

impl Seg2 {
    pub fn new(start: Vec2, end: Vec2) -> Self {
        Self { start, end }
    }

    pub fn delta(&self) -> Vec2 {
        self.end - self.start
    }

    pub fn crosses_x_axis_at(&self) -> Option<Vec2> {
        if (self.start.y >= 0. && self.end.y >= 0.) || (self.start.y <= 0. && self.end.y <= 0.) {
            return None;
        }
        let delta = self.delta();
        let mult = -self.start.y / delta.y;
        Some(self.start + (delta * mult))
    }
}
