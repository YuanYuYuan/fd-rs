use crate::base::Equation;
use num_traits::Float;

// #[derive(Debug, Copy, Clone)]
pub struct Advection<T> {
    pub a: T,
}

impl<T> Equation<T> for Advection<T>
where
    T: Float,
{
    fn f(&self, u: T) -> T {
        u * self.a
    }

    fn df(&self, _u: T) -> T {
        self.a
    }
}


pub struct InviscidBurger;

impl<T> Equation<T> for InviscidBurger
where
    T: Float,
{
    fn f(&self, u: T) -> T {
        u.powi(2) / T::from(2).unwrap()
    }

    fn df(&self, u: T) -> T {
        u
    }
}
