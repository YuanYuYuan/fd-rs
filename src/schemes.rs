use crate::base::Equation;
use crate::base::Simluation;
use itertools::izip;
use ndarray::Array1;
use num_traits::Float;
use std::fmt::Debug;

/// # Scheme
///
/// The scheme currently will only support solving 1D hyperbolic equation.
///
/// $$
/// u_t + f(u)_x = 0
/// $$
///
/// The explicit scheme works like
/// $$
/// u_{j}^{n+1} = u_{j}^{n} - \frac{\Delta t}{\Delta x} (h_{j+}^{n} - h_{j-}^{n}),
/// $$
///
/// where $h_{j+}^{n}, h_{j-}^{n}$ are the numerical flux.

pub trait Scheme<T>: Debug
where
    T: Float,
{
    /// # Spreading speed
    ///
    /// $$
    /// v_{j+} = \begin{cases}
    /// \frac{\Delta t}{\Delta x} \frac{f_{j+1} - f_{j}}{u_{j+1}-u_{j}}
    /// ,& u_{j} != u_{j+1} \\\\
    /// \frac{\Delta t}{\Delta x} f^{'}(u_j),& u_{j} = u_{j+1}
    /// \end{cases}
    /// $$
    ///
    /// $$
    /// v_{j-} = \begin{cases}
    /// \frac{\Delta t}{\Delta x} \frac{f_{j} - f_{j-1}}{u_{j}-u_{j-1}}
    /// ,& u_{j} != u_{j-1} \\\\
    /// \frac{\Delta t}{\Delta x} f^{'}(u_{j-1}),& u_{j} = u_{j-1}
    /// \end{cases}
    /// $$
    ///
    /// The return size = n + ext

    fn speed(&self, sim: &Simluation<T>, eq: &dyn Equation<T>, ext: usize) -> [Array1<T>; 2] {
        let n = sim.len();
        let dt_over_dx = sim.dt_over_dx();

        // extended u: [n+2*(ext+1)]
        let u = sim.get_u(ext + 1);
        let u_iter = u.iter();

        // compute difference of u: [n+2*ext+1]
        let du_iter = u_iter
            .clone()
            .zip(u_iter.clone().skip(1))
            .map(|(&l, &r)| r - l);

        // extended f: [n+2*(ext+1)]
        let f = sim.get_f(eq, ext + 1);
        let f_iter = f.iter();

        // compute difference of f: [n+2*ext+1]
        let df_iter = f_iter
            .clone()
            .zip(f_iter.clone().skip(1))
            .map(|(&l, &r)| r - l);

        // compute v each case
        let compute_v = |(&u, du, df)| {
            let df_du = if du == T::from(0).unwrap() {
                eq.df(u)
            } else {
                df / du
            };
            let v = df_du * dt_over_dx;
            assert!(
                v.abs() <= T::from(1).unwrap(),
                format!("Check the CRL condition! {:?}", v.to_f64().unwrap())
            );
            v
        };

        // v+: [n]
        let v_pos: Vec<T> = izip!(
            u_iter.clone().skip(1),  // u_j
            du_iter.clone().skip(1), // u_{j+1} - u_j
            df_iter.clone().skip(1), // f_{j+1} - f_j
        )
        .map(compute_v)
        .collect();

        // v-: [n]
        let v_neg: Vec<T> = izip!(
            u_iter.clone().skip(ext),  // u_j
            du_iter.take(n + ext * 2), // u_{j} - u_{j-1}
            df_iter.take(n + ext * 2), // f_{j} - f_{j-1}
        )
        .map(compute_v)
        .collect();

        // sanity check
        assert_eq!(v_neg.len(), v_pos.len());
        assert_eq!(v_pos.len(), n + 2 * ext);

        [Array1::<T>::from(v_neg), Array1::<T>::from(v_pos)]
    }

    /// # Conservative Finite Difference Schemes
    ///
    /// $$
    /// u_{j+1} = u_{j} = \frac{\Delta t}{\Delta x} (h_{j+} - h_{j-})
    /// $$

    fn run(&self, sim: &Simluation<T>, eq: &dyn Equation<T>) -> Array1<T> {
        let [h_neg, h_pos] = self.flux(&sim, eq);
        let u = sim.get_u(0);
        let dt_over_dx = sim.dt_over_dx();
        u - (h_pos - h_neg).mapv(|x| dt_over_dx * x)
    }

    fn flux(&self, sim: &Simluation<T>, eq: &dyn Equation<T>) -> [Array1<T>; 2];
}

// pub trait CloneScheme<T> {
//     fn clone_scheme(&self) -> Box<dyn Scheme<T>>;
// }

// impl<S, T> CloneScheme<T> for S
// where
//     S: Scheme<T> + Clone + 'static,
//     T: Float,
// {
//     fn clone_scheme(&self) -> Box<dyn Scheme<T>> {
//         Box::new(self.clone())
//     }
// }

// impl<T> Clone for Box<dyn Scheme<T>> {
//     fn clone(&self) -> Self {
//         self.clone_scheme()
//     }
// }

/// ## Scheme: Upwind
///
/// The numerical flux $h$ of the upwind is determined
/// by the direction of its spreading speed $v$
/// $$
/// h_{j+} = \begin{cases}
/// f_j,& v_{j+} > 0 \\\\
/// f_{j+1},& v_{j+} < 0
/// \end{cases}
/// $$
///
/// $$
/// h_{j-} = \begin{cases}
/// f_{j-1},& v_{j-} > 0 \\\\
/// f_j,& v_{j-} < 0
/// \end{cases}
/// $$

#[derive(Debug, Copy, Clone)]
pub struct Upwind;

impl<T: Float> Scheme<T> for Upwind {
    fn flux(&self, sim: &Simluation<T>, eq: &dyn Equation<T>) -> [Array1<T>; 2] {
        let ext = 1;
        let f = sim.get_f(eq, ext);
        let [v_neg, v_pos] = self.speed(&sim, eq, 0);

        // h_{j+}
        let h_pos: Vec<T> = izip!(
            v_pos.iter(),             // v_{j+}
            f.iter().clone().skip(1), // f_j
            f.iter().clone().skip(2), // f_{j+1}
        )
        .map(
            |(&v, &f, &f_next)| {
                if v > T::from(0).unwrap() {
                    f
                } else {
                    f_next
                }
            },
        )
        .collect();

        // h_{j-}
        let h_neg: Vec<T> = izip!(
            v_neg.iter(),             // v_{j-}
            f.iter().clone().skip(1), // f_j
            f.iter().clone(),         // f_{j-1}
        )
        .map(
            |(&v, &f, &f_prev)| {
                if v < T::from(0).unwrap() {
                    f
                } else {
                    f_prev
                }
            },
        )
        .collect();

        // sanity check
        assert_eq!(h_neg.len(), h_pos.len());
        assert_eq!(h_neg.len(), sim.len());

        [Array1::<T>::from(h_neg), Array1::<T>::from(h_pos)]
    }
}

/// ## Scheme: Beam-Warming
///
/// Like the *upwind*, the numerical flux $h$ of the upwind is determined
/// by the direction of its spreading speed $v$
///
/// $$
/// h_{j+} = \begin{cases}
/// \frac{1}{2} (3f_{j} - f_{j-1}) - \frac{1}{2} v_{j-}(f_{j} - f_{j-1}),& v_{j+} > 0 \\\\
/// \frac{1}{2} (3f_{j+1} - f_{j+2}) - \frac{1}{2} v_{(j+)+1}(f_{j+2} - f_{j+1}),& v_{j+} < 0
/// \end{cases}
/// $$
///
/// $$
/// h_{j-} = \begin{cases}
/// \frac{1}{2} (3f_{j-1} - f_{j-2}) - \frac{1}{2} v_{(j-)-1}(f_{j-1} - f_{j-2}),& v_{j-} > 0 \\\\
/// \frac{1}{2} (3f_{j} - f_{j+1}) - \frac{1}{2} v_{j+}(f_{j+1} - f_{j}),& v_{j-} < 0
/// \end{cases}
/// $$

#[derive(Debug, Copy, Clone)]
pub struct BeamWarming;

impl<T: Float> Scheme<T> for BeamWarming {
    fn flux(&self, sim: &Simluation<T>, eq: &dyn Equation<T>) -> [Array1<T>; 2] {
        // f: [n+4]
        let n = sim.len();
        let ext = 2;
        let f = sim.get_f(eq, ext);

        // v+, v-: [n+2]
        let [v_neg, v_pos] = self.speed(&sim, eq, 1);

        let zero = T::from(0).unwrap();
        let three = T::from(3).unwrap();
        let two = T::from(2).unwrap();

        // h_{j+}
        let h_pos: Vec<T> = izip!(
            v_neg.iter().skip(1),     // v1: v_{j-}
            v_pos.iter().skip(1),     // v2: v_{j+}
            v_pos.iter().skip(2),     // v3: v_{(j+)+1}
            f.iter().clone().skip(1), // f1: f_{j-1}
            f.iter().clone().skip(2), // f2: f_j
            f.iter().clone().skip(3), // f3: f_{j+1}
            f.iter().clone().skip(4), // f4: f_{j+2}
        )
        .map(|(&v1, &v2, &v3, &f1, &f2, &f3, &f4)| {
            if v2 > zero {
                ((three * f2 - f1) - v1 * (f2 - f1)) / two
            } else {
                ((three * f3 - f4) - v3 * (f4 - f3)) / two
            }
        })
        .collect();

        // h_{j-}
        let h_neg: Vec<T> = izip!(
            v_neg.iter().skip(0),     // v1: v_{(j-)-1}
            v_neg.iter().skip(1),     // v2: v_{j-}
            v_pos.iter().skip(1),     // v3: v_{j+}
            f.iter().clone().skip(0), // f1: f_{j-2}
            f.iter().clone().skip(1), // f2: f_{j-1}
            f.iter().clone().skip(2), // f2: f_j
            f.iter().clone().skip(3), // f3: f_{j+1}
        )
        .take(n)
        .map(|(&v1, &v2, &v3, &f1, &f2, &f3, &f4)| {
            if v2 > zero {
                ((three * f2 - f1) - v1 * (f2 - f1)) / two
            } else {
                ((three * f3 - f4) - v3 * (f4 - f3)) / two
            }
        })
        .collect();

        // sanity check
        assert_eq!(h_neg.len(), h_pos.len());
        assert_eq!(h_neg.len(), n);

        [Array1::<T>::from(h_neg), Array1::<T>::from(h_pos)]
    }
}

/// ## Scheme: Lax-Wendroff
///
/// The numerical flux is given by
///
/// $$
/// h_{j+} = f\left(\frac{u_{j+1} +
/// u_{j}}{2} - \frac{\Delta t}{2 \Delta x}
/// (f_{j+1} - f_j)\right)
/// $$
///
/// $$
/// h_{j-} = f\left(\frac{u_{j} +
/// u_{j-1}}{2} - \frac{\Delta t}{2 \Delta x}
/// (f_{j} - f_{j-1})\right)
/// $$

#[derive(Debug, Copy, Clone)]
pub struct LaxWendroff;

impl<T: Float> Scheme<T> for LaxWendroff {
    fn flux(&self, sim: &Simluation<T>, eq: &dyn Equation<T>) -> [Array1<T>; 2] {
        let ext = 1;
        let dt_over_dx = sim.dt_over_dx();
        let n = sim.len();

        // extended u: [n+2]
        let u = sim.get_u(ext);
        let u_iter = u.iter();

        // extended f: [n+2]
        let f = sim.get_f(eq, ext);
        let f_iter = f.iter();

        // h_{j+}
        let h_pos: Vec<T> = izip!(
            u_iter.clone().skip(1), // u_{j}
            u_iter.clone().skip(2), // u_{j+1}
            f_iter.clone().skip(1), // f_{j}
            f_iter.clone().skip(2), // f_{j+1}
        )
        .map(|(&u, &u_next, &f, &f_next)| {
            eq.f((u_next + u - dt_over_dx * (f_next - f)) / T::from(2).unwrap())
        })
        .collect();

        // h_{j-}
        let h_neg: Vec<T> = izip!(
            u_iter.clone().skip(1), // u_{j}
            u_iter.clone().take(n), // u_{j-1}
            f_iter.clone().skip(1), // f_{j}
            f_iter.clone().take(n), // f_{j-1}
        )
        .map(|(&u, &u_prev, &f, &f_prev)| {
            eq.f((u + u_prev - dt_over_dx * (f - f_prev)) / T::from(2).unwrap())
        })
        .collect();

        // sanity check
        assert_eq!(h_neg.len(), h_pos.len());
        assert_eq!(h_neg.len(), sim.len());

        [Array1::<T>::from(h_neg), Array1::<T>::from(h_pos)]
    }
}

/// ## Scheme: Lax-Friedrichs
///
/// The numerical flux is given by
///
/// $$
/// h_{j+} = \frac{1}{2}(f_{j+1} + f_{j}) - \frac{\Delta x}{2 \Delta t}(u_{j+1} - u_{j})
/// $$
///
/// $$
/// h_{j-} = \frac{1}{2}(f_{j} + f_{j-1}) - \frac{\Delta x}{2 \Delta t}(u_{j} - u_{j-1})
/// $$

#[derive(Debug, Copy, Clone)]
pub struct LaxFriedrichs;

impl<T: Float> Scheme<T> for LaxFriedrichs {
    fn flux(&self, sim: &Simluation<T>, eq: &dyn Equation<T>) -> [Array1<T>; 2] {
        let ext = 1;
        let dt_over_dx = sim.dt_over_dx();
        let n = sim.len();

        // extended u: [n+2]
        let u = sim.get_u(ext);
        let u_iter = u.iter();

        // extended f: [n+2]
        let f = sim.get_f(eq, ext);
        let f_iter = f.iter();

        // h_{j+}
        let h_pos: Vec<T> = izip!(
            u_iter.clone().skip(1), // u_{j}
            u_iter.clone().skip(2), // u_{j+1}
            f_iter.clone().skip(1), // f_{j}
            f_iter.clone().skip(2), // f_{j+1}
        )
        .map(|(&u, &u_next, &f, &f_next)| {
            ((f_next + f) - dt_over_dx * (u_next - u)) / T::from(2).unwrap()
        })
        .collect();

        // h_{j-}
        let h_neg: Vec<T> = izip!(
            u_iter.clone().skip(1), // u_{j}
            u_iter.clone().take(n), // u_{j-1}
            f_iter.clone().skip(1), // f_{j}
            f_iter.clone().take(n), // f_{j-1}
        )
        .map(|(&u, &u_prev, &f, &f_prev)| {
            ((f + f_prev) - dt_over_dx * (u - u_prev)) / T::from(2).unwrap()
        })
        .collect();

        // sanity check
        assert_eq!(h_neg.len(), h_pos.len());
        assert_eq!(h_neg.len(), sim.len());

        [Array1::<T>::from(h_neg), Array1::<T>::from(h_pos)]
    }
}
