import numpy as np
import openmdao.api as om

class SellarDis1(om.ExplicitComponent):
    """
    Component containing Discipline 1 -- no derivatives version.
    """

    def setup(self):
        # ****************************
        # TODO: define inputs and outputs
        

        pass
        # ****************************

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        """
        Evaluates the equation
        y1 = z1**2 + z2 + x1 - 0.2*y2
        """
        # ****************************
        # TODO: implement the model
        

        pass
        # ****************************


class SellarDis2(om.ExplicitComponent):
    """
    Component containing Discipline 2 -- no derivatives version.
    """

    def setup(self):
        # Global Design Variable
        self.add_input('z', val=np.zeros(2))

        # Coupling parameter
        self.add_input('y1', val=1.0)

        # Coupling output
        self.add_output('y2', val=1.0)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        """
        Evaluates the equation
        y2 = y1**(.5) + z1 + z2
        """

        z1 = inputs['z'][0]
        z2 = inputs['z'][1]
        y1 = inputs['y1']

        # Note: this may cause some issues. However, y1 is constrained to be
        # above 3.16, so lets just let it converge, and the optimizer will
        # throw it out
        if y1.real < 0.0:
            y1 *= -1

        outputs['y2'] = y1**.5 + z1 + z2


if __name__ == '__main__':
    # setup model
    prob = om.Problem()
    prob.model.add_subsystem('d1', SellarDis1())
    prob.model.add_subsystem('d2', SellarDis2())
    # run model with default input values defined in setup()
    prob.setup()
    prob.run_model()
    # check model outputs
    print('y1 =', prob.get_val('d1.y1'))
    print('y2 =', prob.get_val('d2.y2'))
    np.testing.assert_allclose(prob.get_val('d1.y1'), -0.2, rtol=1e-7)  # check y1 output
    np.testing.assert_allclose(prob.get_val('d2.y2'), 1.0, rtol=1e-7)  # check y2 output
