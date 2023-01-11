import openmdao.api as om
from sellar_group import SellarMDA

if __name__ == '__main__':
    # add sellar MDA model to the problem
    prob = om.Problem()
    prob.model = SellarMDA()

    # define optimizer
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    # prob.driver.options['maxiter'] = 100
    prob.driver.options['tol'] = 1e-8

    # ********************************************
    # TODO: define optimization problem (design variables, objective, constraints)
    prob.model.add_design_var('x', lower=0, upper=10)
    prob.model.add_design_var('z', lower=0, upper=10)
    prob.model.add_objective('obj')
    prob.model.add_constraint('con1', upper=0)
    prob.model.add_constraint('con2', upper=0)
    # ********************************************

    # Ask OpenMDAO to finite-difference across the model to compute the gradients for the optimizer
    ### prob.model.approx_totals()

    # setup OpenMDAO problem
    prob.setup()
    prob.set_solver_print(level=0)  # turn off nonlinear solver outputs

    # run optimization
    prob.run_driver()

    print('minimum found at')
    print('x =', prob.get_val('x')[0])
    print('z =', prob.get_val('z'))

    print('minumum objective')
    print('f =', prob.get_val('obj')[0])