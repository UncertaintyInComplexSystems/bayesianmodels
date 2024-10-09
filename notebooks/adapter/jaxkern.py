from gpjax import kernels

class RBF():

    def __init__(self) -> None:
        print('init overwriter jaxkern RBF')

    def cross_covariance(self, params: dict, x, y):
        return kernels.RBF(
            lengthscale=params['lengthscale'], 
            variance=params['variance']).cross_covariance(x, x)