import numpy as np
import openmdao.api as om
from sellar_components_sol import SellarDis1, SellarDis2

class SellarMDA(om.Group):
    """
    Group for SellarDis1-SellarDis2 coupled model.
    """

    def setup(self):
        cycle = self.add_subsystem('cycle', om.Group(), promotes=['*'])
        cycle.add_subsystem('d1', SellarDis1(), promotes_inputs=['x', 'z'])
        cycle.add_subsystem('d2', SellarDis2(), promotes_inputs=['z'])
        cycle.connect('d1.y1', 'd2.y1')   # connect y1 (output of discipline 1) to input of discipline 2
        cycle.connect('d2.y2', 'd1.y2')   # connect y2 (output of discipline 2) to input of discipline 1

        # set initial design variable values
        cycle.set_input_defaults('x', 1.0)
        cycle.set_input_defaults('z', np.array([5.0, 2.0]))

        # add nonlinear solver (Nonlinear Block Gauss Seidel) to solve the d1-d2 coupling
        cycle.nonlinear_solver = om.NonlinearBlockGS()

        # compute objective and constraints
        self.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                  z=np.array([0.0, 0.0]), x=0.0),
                           promotes=['x', 'z', 'obj'])
        self.connect('d1.y1', 'obj_cmp.y1')
        self.connect('d2.y2', 'obj_cmp.y2')

        self.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'), promotes=['con1'])
        self.connect('d1.y1', 'con_cmp1.y1')

        self.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'), promotes=['con2'])
        self.connect('d2.y2', 'con_cmp2.y2')


if __name__ == '__main__':
    # setup problem
    prob = om.Problem()
    prob.model.add_subsystem('model', SellarMDA(), promotes=['*'])
    prob.setup(check=True)

    # visualize model structure
    om.n2(prob)

    # run nonlinear solver
    prob.run_model()

    # print results
    print('y1 =', prob.get_val('d1.y1'))
    print('y2 =', prob.get_val('d2.y2'))
