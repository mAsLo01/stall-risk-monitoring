import numpy as np


class TurbulenceModel:
    """
    Коррелированные флуктуации для постсрывного режима.

    Это не модель атмосферы, а учебная модель неустойчивости
    аэродинамических коэффициентов после отрыва потока.
    """

    def __init__(self, sigma_cl=0.18, sigma_cd=0.08, sigma_cm=0.1, tau=0.45):
        self.sigma_cl = sigma_cl # амплитуда шума для Cl
        self.sigma_cd = sigma_cd # амплитиуда шума для Cd
        self.sigma_cm = sigma_cm # амплитуда шума для Cm
        self.tau = tau # время корреляции шума: чем больше tau, тем более плавные и медленные колебания

        self.cl_noise = 0.0
        self.cd_noise = 0.0
        self.cm_noise = 0.0

    def _ou_step(self, x, sigma, dt):
        """
        Один шаг процесса Орнштейна–Уленбека.
        Даёт плавный шум, а не дёрганый белый шум.
        -x / self.tau - первая часть - тянет шум обратно к нулю
        sigma * np.sqrt(dt) * np.random.randn() - вторая часть - добавляет случайный толчок
        """

        dx = -x / self.tau * dt + sigma * np.sqrt(dt) * np.random.randn()
        return x + dx

    def update(self, sep, dt):
        """
        Возвращает флуктуации dCL, dCD, dCm.

        Амплитуда растёт вместе с sep.
        """

        self.cl_noise = self._ou_step(self.cl_noise, self.sigma_cl, dt)
        self.cd_noise = self._ou_step(self.cd_noise, self.sigma_cd, dt)
        self.cm_noise = self._ou_step(self.cm_noise, self.sigma_cm, dt)

        intensity = sep

        dCL = intensity * self.cl_noise
        dCD = intensity * abs(self.cd_noise)
        dCm = intensity * self.cm_noise

        return dCL, dCD, dCm