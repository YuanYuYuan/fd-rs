use clap::Clap;
use fdm::base::Simluation;
use fdm::equations::{Advection, InviscidBurger};
use fdm::schemes::{BeamWarming, LaxFriedrichs, LaxWendroff, Upwind};
use fdm::{BoxedEquation, BoxedFunction, BoxedScheme};
use gnuplot::{AxesCommon, Figure, Fix, Font};
use itertools::iproduct;
use rayon::prelude::*;
use std::f64::consts::PI;
use std::fs;

#[derive(Clap)]
struct Args {
    #[clap(short, long, default_value = "outputs")]
    output_dir: String,
}

#[derive(Debug)]
struct Name {
    equ: String,
    ini: String,
    sch: String,
}

pub struct Domain {
    dx: f64,
    dt: f64,
    time: f64,
    space: [f64; 2],
}

struct Experiment<'a> {
    name: Name,
    equ: &'a BoxedEquation,
    ini: &'a BoxedFunction,
    sch: &'a BoxedScheme,
}

impl Experiment<'_> {
    fn run(&self, output_dir: &str, domain: &Domain) {
        let mut fig = Figure::new();
        let name = format!("{}-{}-{}", self.name.equ, self.name.ini, self.name.sch);
        println!("Processing {}", name);
        fig.set_title(&name).set_terminal(
            "gif animate optimize delay 2 size 480,360",
            &format!("{}/{}.gif", output_dir, name),
        );

        let mut sim = Simluation::<f64>::new(domain.dx, domain.dt, domain.space, self.ini);

        for i in 0..(domain.time / domain.dt) as i32 {
            if i > 0 {
                fig.new_page();
            }
            let ax = fig
                .axes2d()
                .set_title(&name, &[Font("Times", 20.0)])
                .set_x_grid(true)
                .set_y_grid(true)
                .set_y_range(Fix(-1.5), Fix(1.5))
                .set_x_range(Fix(domain.space[0]), Fix(domain.space[1]));

            sim.set_state(self.sch.run(&sim, &**self.equ));
            ax.lines(&sim.grid, &sim.state, &[]);
        }

        fig.show().unwrap();
    }
}

fn main() {
    let args = Args::parse();
    fs::create_dir_all(&args.output_dir).unwrap();

    // conditions
    let dx = 1e-2;
    let cfl = 0.6;
    let dt = cfl * dx;

    let domain = Domain {
        dx,
        dt,
        space: [-3., 3.],
        time: 3.,
    };

    // equations
    let equations: Vec<(&str, BoxedEquation)> = vec![
        ("Advection", Box::new(Advection { a: 1.0 })),
        ("InviscidBurger", Box::new(InviscidBurger)),
    ];

    // initial waves
    let inits: Vec<(&str, BoxedFunction)> = vec![
        ("Sine", Box::new(|x: f64| (PI * x).sin())),
        (
            "Square",
            Box::new(|x: f64| if x >= 0. && x <= 1. { 1.0 } else { 0. }),
        ),
    ];

    // schemes
    let schemes: Vec<(&str, BoxedScheme)> = vec![
        ("Upwind", Box::new(Upwind)),
        ("BeamWarming", Box::new(BeamWarming)),
        ("LaxWendroff", Box::new(LaxWendroff)),
        ("LaxFriedrichs", Box::new(LaxFriedrichs)),
    ];

    let exps: Vec<Experiment> = iproduct!(equations.iter(), inits.iter(), schemes.iter())
        .map(|(equ, ini, sch)| Experiment {
            name: Name {
                equ: (equ.0).into(),
                ini: (ini.0).into(),
                sch: (sch.0).into(),
            },
            equ: &equ.1,
            ini: &ini.1,
            sch: &sch.1,
        })
        .collect();

    exps.into_par_iter()
        .for_each(|exp| exp.run(&args.output_dir, &domain));
}
