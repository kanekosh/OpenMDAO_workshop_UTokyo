import openmdao.api as om


class Paraboloid(om.ExplicitComponent):
    """
    Evaluates the equation f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3.
    """

    def setup(self):
        self.add_input('x', val=0.0)
        self.add_input('y', val=0.0)

        self.add_output('f_xy', val=0.0)

    def setup_partials(self):
        # This time, use user-defined partial derivative instead of finite difference
        self.declare_partials('*', '*', method='exact')

    def compute(self, inputs, outputs):
        """
        f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3

        Minimum at: x = 6.6667; y = -7.3333
        """
        x = inputs['x']
        y = inputs['y']

        outputs['f_xy'] = (x - 3.0)**2 + x * y + (y + 4.0)**2 - 3.0

    def compute_partials(self, inputs, partials):
        """
        Partial derivatives of f(x, y) w.r.t. x and y
        """
        x = inputs['x']
        y = inputs['y']

        # ***********************************
        # TODO: implement partial derivatives
        df_dx = 2 * (x - 3) + y
        df_dy = x + 2 * (y + 4)
        # ***********************************

        partials['f_xy', 'x'] = df_dx
        partials['f_xy', 'y'] = df_dy


if __name__ == "__main__":

    # --- setup the model and OpenMDAO problem ---
    model = om.Group()
    model.add_subsystem('parab_comp', Paraboloid())

    prob = om.Problem(model)
    prob.setup()

    # --- set x and y value ---
    prob.set_val('parab_comp.x', 3.0)
    prob.set_val('parab_comp.y', -4.0)

    # --- evaluate the model (function) and print the output ---
    prob.run_model()
    print(prob['parab_comp.f_xy'])

    # --- verify user-defined derivatives against the finite difference ---
    prob.check_partials(compact_print=True)