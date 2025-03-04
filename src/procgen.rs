use bevy::math::Vec2;
use direction::{CardinalDirection, CardinalDirections, Direction, Directions, DirectionsCardinal};
use grid_2d::{Coord, Grid, Size};
use rand::Rng;
use std::{
    collections::{HashSet, VecDeque},
    mem,
};

use crate::coord_to_vec;

pub struct Map1 {
    pub grid: Grid<bool>,
}

fn blob_to_outside_directions(blob: &HashSet<Coord>) -> (Coord, Vec<CardinalDirection>) {
    assert!(!blob.is_empty());
    let in_blob = |coord| blob.contains(&coord);
    // Find a cell with nothing above it or to its left.
    let start = *blob
        .iter()
        .filter(|&&coord| {
            !in_blob(coord + Direction::North.coord()) && !in_blob(coord + Direction::West.coord())
        })
        .min_by_key(|coord| coord.y)
        .unwrap();
    // The starting cell has nothing above it so it has an outside edge on its top. We'll start by
    // moving east along this edge.
    let mut current_direction = CardinalDirection::East;
    let mut current = start;
    let mut ret = vec![current_direction];
    loop {
        if in_blob(current + current_direction.left45().coord()) {
            current += current_direction.left45().coord();
            current_direction = current_direction.left90();
        } else if in_blob(current + current_direction.coord()) {
            current += current_direction.coord();
            // current_direction is unchanged
        } else {
            // current is unchanged
            current_direction = current_direction.right90();
        }
        // we've made it back to the start
        if current == start && current_direction == CardinalDirection::East {
            break;
        }
        ret.push(current_direction);
    }
    (start, ret)
}

fn cardinal_direction_to_unit_vec2(direction: CardinalDirection) -> Vec2 {
    match direction {
        CardinalDirection::North => Vec2::new(0., -1.),
        CardinalDirection::East => Vec2::new(1., 0.),
        CardinalDirection::South => Vec2::new(0., 1.),
        CardinalDirection::West => Vec2::new(-1., 0.),
    }
}

fn cardinal_directions_to_linestrip(start: Coord, directions: &[CardinalDirection]) -> Vec<Vec2> {
    let mut grouped_directions = Vec::new();
    for &d in directions {
        if let Some(&mut (ref last_d, ref mut count)) = grouped_directions.last_mut() {
            if *last_d == d {
                *count += 1;
                continue;
            }
        }
        grouped_directions.push((d, 1));
    }
    let mut cursor = coord_to_vec(start);
    let mut ret = vec![cursor];
    for (d, count) in grouped_directions {
        let v = cardinal_direction_to_unit_vec2(d) * count as f32;
        cursor += v;
        ret.push(cursor);
    }
    ret
}

impl Map1 {
    pub fn new() -> Self {
        Self {
            grid: Grid::new_copy(Size::new(30, 20), false),
        }
    }

    pub fn new_with_size(size: Size) -> Self {
        Self {
            grid: Grid::new_copy(size, true),
        }
    }

    fn generate_cave<R: Rng>(&mut self, rng: &mut R) {
        let num_steps = 4;
        let num_clean_steps = 10;
        let survive_min = 4;
        let survive_max = 8;
        let resurrect_min = 5;
        let resurrect_max = 5;
        let mut next = self.grid.clone();
        loop {
            // initialize to noise
            for cell in self.grid.iter_mut() {
                *cell = rng.random();
            }
            // generate basic shape with a cell automata
            for _ in 0..num_steps {
                for ((coord, alive), next_alive) in self.grid.enumerate().zip(next.iter_mut()) {
                    if self.grid.is_on_edge(coord) {
                        *next_alive = true;
                        continue;
                    }
                    let mut alive_neighbour_count = 0;
                    for d in Directions {
                        if let Some(true) = self.grid.get(coord + d.coord()) {
                            alive_neighbour_count += 1;
                        }
                    }
                    *next_alive = if *alive {
                        alive_neighbour_count >= survive_min && alive_neighbour_count <= survive_max
                    } else {
                        alive_neighbour_count >= resurrect_min
                            && alive_neighbour_count <= resurrect_max
                    };
                }
                mem::swap(&mut self.grid, &mut next);
            }
            // remove cells with only one neighbour repeating several times
            for _ in 0..num_clean_steps {
                for ((coord, alive), next_alive) in self.grid.enumerate().zip(next.iter_mut()) {
                    if self.grid.is_on_edge(coord) {
                        continue;
                    }
                    let mut alive_neighbour_count = 0;
                    for d in Directions {
                        if let Some(true) = self.grid.get(coord + d.coord()) {
                            alive_neighbour_count += 1;
                        }
                    }
                    *next_alive = if *alive {
                        alive_neighbour_count >= 2
                    } else {
                        alive_neighbour_count >= 5
                    };
                }
            }
            // find the largest contiguous open area
            let mut largest = HashSet::new();
            let mut largest_candidate = HashSet::new();
            let mut seen = HashSet::new();
            let mut to_visit = VecDeque::new();
            for (coord, &alive) in self.grid.enumerate() {
                if alive || !seen.insert(coord) {
                    continue;
                }
                to_visit.push_back(coord);
                largest_candidate.clear();
                largest_candidate.insert(coord);
                while let Some(coord) = to_visit.pop_front() {
                    for d in DirectionsCardinal {
                        let neighbour_coord = coord + d.coord();
                        if self.grid.get(neighbour_coord) == Some(&false)
                            && seen.insert(neighbour_coord)
                        {
                            to_visit.push_back(neighbour_coord);
                            largest_candidate.insert(neighbour_coord);
                        }
                    }
                }
                if largest_candidate.len() > largest.len() {
                    mem::swap(&mut largest, &mut largest_candidate);
                }
            }
            for (coord, alive) in self.grid.enumerate_mut() {
                *alive = !largest.contains(&coord);
            }
            if largest.len() < 250 {
                continue;
            }
            log::info!("Open space made up of {} cells.", largest.len());
            break;
        }
    }

    pub fn generate<R: Rng>(&mut self, rng: &mut R) {
        self.generate_cave(rng);
    }

    fn open_blob(&self) -> HashSet<Coord> {
        let start = self
            .grid
            .enumerate()
            .find(|&(_coord, &alive)| !alive)
            .unwrap()
            .0;
        let mut blob = HashSet::new();
        blob.insert(start);
        let mut to_visit = VecDeque::new();
        to_visit.push_back(start);
        while let Some(coord) = to_visit.pop_front() {
            for d in CardinalDirections {
                let neighbour_coord = coord + d.coord();
                if self.grid.get(neighbour_coord) == Some(&false) && blob.insert(neighbour_coord) {
                    to_visit.push_back(neighbour_coord);
                }
            }
        }
        blob
    }

    fn floating_blobs(&self) -> Vec<HashSet<Coord>> {
        let mut seen = HashSet::new();
        let mut ret = Vec::new();
        for (coord, alive) in self.grid.enumerate() {
            if !seen.insert(coord) || !alive {
                continue;
            }
            let mut blob = HashSet::new();
            blob.insert(coord);
            let mut to_visit = VecDeque::new();
            to_visit.push_back(coord);
            while let Some(coord) = to_visit.pop_front() {
                seen.insert(coord);
                for d in CardinalDirections {
                    let neighbour_coord = coord + d.coord();
                    if blob.insert(neighbour_coord) && self.grid.get(neighbour_coord) == Some(&true)
                    {
                        to_visit.push_back(neighbour_coord);
                    }
                }
            }
            if !blob.contains(&Coord::new(0, 0)) {
                log::info!("adding blob of size {}", blob.len());
                ret.push(blob);
            } else {
                log::info!("skipping non-floating blob of size {}", blob.len());
            }
        }
        ret
    }

    fn carve_out_open_space_from(&mut self, other: &Self, offset: Coord) {
        for (c, open) in other.grid.enumerate() {
            if !*open {
                *self.grid.get_checked_mut(c + offset) = false;
            }
        }
    }

    pub fn make_full<R: Rng>(rng: &mut R) -> (Self, Coord) {
        let dungeon_size = Size::new(30, 30);
        let corridor_offset_x = 0;
        let corridor_offset_y = 10;
        let hub_size = Size::new(8, 8);
        let total_size = Size::new(
            (dungeon_size.width() + corridor_offset_x) * 2 + hub_size.width(),
            dungeon_size.height() * 2,
        );
        let hub_centre = Size::new(
            total_size.width() / 2,
            dungeon_size.height() + corridor_offset_y,
        )
        .to_coord()
        .unwrap();
        let hub_top_left = hub_centre - hub_size.to_coord().unwrap() / 2;
        let full_map = 'outer: loop {
            let dungeons = (0..3)
                .map(|_| {
                    let mut map = Self::new_with_size(dungeon_size);
                    map.generate(rng);
                    map
                })
                .collect::<Vec<_>>();
            let mut full_map = Self::new_with_size(total_size);
            for c in hub_size.coord_iter_row_major() {
                *full_map.grid.get_checked_mut(c + hub_top_left) = false;
            }
            full_map.carve_out_open_space_from(
                &dungeons[0],
                Coord::new(0, dungeon_size.height() as i32),
            );
            full_map.carve_out_open_space_from(
                &dungeons[2],
                Coord::new(
                    (total_size.width() - dungeon_size.width()) as i32,
                    dungeon_size.height() as i32,
                ),
            );
            full_map.carve_out_open_space_from(
                &dungeons[1],
                Coord::new((total_size.width() - dungeon_size.width()) as i32 / 2, 0),
            );
            let mut corridor_cursor = hub_centre - Coord::new(hub_size.width() as i32 / 2 + 1, 1);
            loop {
                match full_map.grid.get_mut(corridor_cursor) {
                    None => continue 'outer,
                    Some(b) => {
                        let stop = !(*b);
                        *b = false;
                        *full_map
                            .grid
                            .get_checked_mut(corridor_cursor + Coord::new(0, 1)) = false;
                        if stop {
                            break;
                        }
                        corridor_cursor += Coord::new(-1, 0);
                    }
                }
            }
            let mut corridor_cursor = hub_centre + Coord::new(hub_size.width() as i32 / 2, -1);
            loop {
                match full_map.grid.get_mut(corridor_cursor) {
                    None => continue 'outer,
                    Some(b) => {
                        let stop = !(*b);
                        *b = false;
                        *full_map
                            .grid
                            .get_checked_mut(corridor_cursor + Coord::new(0, 1)) = false;
                        if stop {
                            break;
                        }
                        corridor_cursor += Coord::new(1, 0);
                    }
                }
            }
            let mut corridor_cursor = hub_centre - Coord::new(1, hub_size.height() as i32 / 2 + 1);
            loop {
                match full_map.grid.get_mut(corridor_cursor) {
                    None => continue 'outer,
                    Some(b) => {
                        let stop = !(*b);
                        *b = false;
                        *full_map
                            .grid
                            .get_checked_mut(corridor_cursor + Coord::new(1, 0)) = false;
                        if stop {
                            break;
                        }
                        corridor_cursor += Coord::new(0, -1);
                    }
                }
            }
            break full_map;
        };
        (full_map, hub_centre)
    }

    pub fn to_map2(&self) -> Map2 {
        let mut wall_strips = Vec::new();
        let (start, open_directions) = blob_to_outside_directions(&self.open_blob());
        wall_strips.push(cardinal_directions_to_linestrip(start, &open_directions));
        for blob in self.floating_blobs() {
            let (start, ds) = blob_to_outside_directions(&blob);
            wall_strips.push(cardinal_directions_to_linestrip(start, &ds));
        }
        /*
        let wall_strips = vec![vec![
            Vec2::new(0., 0.),
            Vec2::new(10., 0.),
            Vec2::new(10., 10.),
            Vec2::new(20., 10.),
        ]];*/
        /*let wall_strips = vec![vec![
            Vec2::new(0., 0.),
            Vec2::new(10., 0.),
            //  Vec2::new(10., 10.),
            //  Vec2::new(0., 10.),
            //  Vec2::new(0., 0.),
        ]];*/
        Map2 { wall_strips }
    }
}

pub struct Map2 {
    pub wall_strips: Vec<Vec<Vec2>>,
}

impl Map2 {
    pub fn new() -> Self {
        Self {
            wall_strips: Vec::new(),
        }
    }
}
