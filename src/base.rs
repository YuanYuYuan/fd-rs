use gnuplot::{AxesCommon, Figure};
use ndarray::{prelude::*, Array1};
use num_traits::Float;
use std::fmt::Debug;

pub struct Simluation<T> {
    pub state: Array1<T>,
    dt: T,
    dx: T,
    pub grid: Array1<T>,
    boundary: Option<[T; 2]>,
}

pub trait Equation<T>: Debug {
    fn f(&self, u: T) -> T;
    fn df(&self, u: T) -> T;
}

impl<T> Default for Simluation<T>
where
    T: Float,
{
    fn default() -> Self {
        let dx = T::from(1e-2).unwrap();
        let cfl = T::from(0.6).unwrap();
        let space = Array::range(T::from(-5).unwrap(), T::from(5).unwrap(), dx);
        let n = space.len();
        Self {
            dt: cfl * dx,
            dx: dx,
            state: Array1::<T>::zeros(n),
            grid: space,
            boundary: None,
        }
    }
}

impl<T> Simluation<T>
where
    T: Float,
{
    pub fn len(&self) -> usize {
        self.state.len()
    }

    pub fn set_state(&mut self, new_state: Array1<T>) {
        assert_eq!(self.len(), new_state.len());
        self.state = new_state;
    }

    pub fn new<F>(dx: T, dt: T, range: [T; 2], init: F) -> Self
    where
        F: Fn(T) -> T,
    {
        let grid = Array::range(range[0], range[1], dx);
        let state = grid.mapv(init);
        Self {
            dx: dx,
            dt: dt,
            // boundary: Some([T::from(1.).unwrap(), T::from(1.).unwrap()]),
            boundary: None,
            grid: grid,
            state: state,
        }
    }

    pub fn dt_over_dx(&self) -> T {
        self.dt / self.dx
    }

    // get discrete u
    pub fn get_u(&self, ext: usize) -> Array1<T> {
        let u = if ext > 0 {
            let u = &self.state;
            let mut v: Vec<T> = u.to_vec();

            for i in 0..ext {
                // left boundary
                v.insert(
                    0,
                    match self.boundary {
                        Some(b) => b[0],            // left source
                        None => u[u.len() - 1 - i], // loop to the right
                    },
                );

                // right boundary
                v.push(match self.boundary {
                    Some(b) => b[1], // right source
                    None => u[i],    // loop to the left
                });
            }

            Array1::<T>::from(v)
        } else {
            self.state.clone()
        };

        // sanity check
        assert_eq!(self.len() + 2 * ext, u.len());
        u
    }

    // get discrete f
    pub fn get_f(&self, eq: &dyn Equation<T>, ext: usize) -> Array1<T> {
        let f = self.get_u(ext).map(|x| eq.f(*x));

        // sanity check
        assert_eq!(self.len() + 2 * ext, f.len());
        f
    }

    pub fn plot(&self, name: &str) {
        let mut fg = Figure::new();

        // Convert to f64 since gnuplot only support this
        let grid: Array1<f64> = self.grid.map(|x| x.to_f64().unwrap());
        let state: Array1<f64> = self.state.map(|x| x.to_f64().unwrap());
        fg.set_title(&name).set_offset(2.0, 0.0);
        fg.axes2d()
            // .set_aspect_ratio(AutoOption::Fix(0.5))
            // .set_size(0.6, 0.4)
            .set_x_grid(true)
            .set_y_grid(true)
            .lines(&grid, &state, &[]);
        fg.show().unwrap();
    }
}
