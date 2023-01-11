"""
Solve a toy optimization problem:
minimize:   f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3
w.r.t.:     x, y
subject to: 0 <= x + y <= 10
"""


# We'll use the component that was defined in the last tutorial
from paraboloid_2_sol import Paraboloid

import openmdao.api as om

if __name__ == '__main__':
    # build the model
    prob = om.Problem()
    prob.model.add_subsystem('parab', Paraboloid(), promotes_inputs=['x', 'y'])

    # define the component to compute the constraint function
    prob.model.add_subsystem('const', om.ExecComp('g = x + y'), promotes_inputs=['x', 'y'])

    # Set initial values for the design variables 'x' and 'y'
    prob.model.set_input_defaults('x', 3.0)
    prob.model.set_input_defaults('y', -4.0)

    # setup optimizer (scipy's SLSQP)
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'

    # define optimization variables and bounds
    prob.model.add_design_var('x', lower=-50, upper=50)
    prob.model.add_design_var('y', lower=-50, upper=50)

    # define objective function to be minimized
    prob.model.add_objective('parab.f_xy')

    # define constraint: 0 <= g <= 10
    prob.model.add_constraint('const.g', lower=0, upper=10.)

    # solve optimization
    prob.setup()
    prob.run_driver()

    # print results
    print('x =', prob.get_val('x'))
    print('y =', prob.get_val('y'))
    print('f =', prob.get_val('parab.f_xy'))
    print('g =', prob.get_val('const.g'))
