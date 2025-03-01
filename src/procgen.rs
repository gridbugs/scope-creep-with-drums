use direction::{Directions, DirectionsCardinal};
use grid_2d::{Coord, Grid, Size};
use rand::Rng;
use std::{
    collections::{HashSet, VecDeque},
    mem,
};

pub struct Map1 {
    pub grid: Grid<bool>,
}

impl Map1 {
    pub fn new() -> Self {
        Self {
            grid: Grid::new_copy(Size::new(30, 20), false),
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
            if largest.len() < 200 {
                continue;
            }
            break;
        }
    }

    pub fn generate<R: Rng>(&mut self, rng: &mut R) {
        self.generate_cave(rng);
    }

    fn linestrips(&self) {
        //todo
    }
}
