use crate::base::Equation;
use crate::base::Simluation;
use itertools::izip;
use ndarray::Array1;
use num_traits::Float;

pub trait Scheme<T: Float> {
    fn speed<E: Equation<T>>(&self, sim: &Simluation<T>, eq: &E) -> [Array1<T>; 2] {
        let extend = true;
        let n = sim.len();
        let dt_over_dx = sim.dt_over_dx();

        // extended u: [n+2]
        let u = sim.get_u(extend);
        let u_iter = u.iter();

        // compute difference of u: [n+1]
        let du_iter = u_iter
            .clone()
            .zip(u_iter.clone().skip(1))
            .map(|(&l, &r)| r - l);

        // extended f: [n+2]
        let f = sim.get_f(eq, extend);
        let f_iter = f.iter();

        // compute difference of f: [n+1]
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
            u_iter.clone().skip(1), // u_j
            du_iter.take(n),        // u_{j} - u_{j-1}
            df_iter.take(n),        // f_{j} - f_{j-1}
        )
        .map(compute_v)
        .collect();

        [Array1::<T>::from(v_neg), Array1::<T>::from(v_pos)]
    }

    fn run<E: Equation<T>>(&self, sim: &Simluation<T>, eq: &E) -> Array1<T> {
        let [h_neg, h_pos] = self.flux(&sim, eq);
        let u = sim.get_u(false);
        let dt_over_dx = sim.dt_over_dx();
        u - (h_pos - h_neg).mapv(|x| dt_over_dx * x)
    }

    fn flux<E: Equation<T>>(&self, sim: &Simluation<T>, eq: &E) -> [Array1<T>; 2];
}

/// ## Scheme: Upwind
///
/// The numerical flux $h$ of the upwind is determined
/// by the direction of its spreading speed $v$
///

pub struct Upwind;

impl<T: Float> Scheme<T> for Upwind {
    fn flux<E: Equation<T>>(&self, sim: &Simluation<T>, eq: &E) -> [Array1<T>; 2] {
        let extend = true;
        let f = sim.get_f(eq, extend);
        let [v_neg, v_pos] = self.speed(&sim, eq);

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

pub struct LaxWendroff;

impl<T: Float> Scheme<T> for LaxWendroff {
    fn flux<E: Equation<T>>(&self, sim: &Simluation<T>, eq: &E) -> [Array1<T>; 2] {
        let extend = true;
        let dt_over_dx = sim.dt_over_dx();
        let n = sim.len();

        // extended u: [n+2]
        let u = sim.get_u(extend);
        let u_iter = u.iter();

        // extended f: [n+2]
        let f = sim.get_f(eq, extend);
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

        [Array1::<T>::from(h_neg), Array1::<T>::from(h_pos)]
    }
}
