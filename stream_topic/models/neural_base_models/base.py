from overrides import EnforceOverrides, overrides


class BaseNN(EnforceOverrides):

    def get_beta():
        pass

    def get_theta():
        pass


class ChildClass(BaseNN):

    @overrides
    def get_beta():
        pass

    @overrides
    def get_theta():
        pass

    def gamma(self):
        pass
