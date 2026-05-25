import numpy as np

class Atmosphere:
    """
    Модель стандартной атмосферы.
    Используется для расчёта плотности воздуха в зависимости от высоты.
    """

    def __init__(self):
        self.rho0 = 1.225  # кг/м³ на уровне моря
        self.H = 8500      # масштаб высоты атмосферы (м)

    def density(self, h: float) -> float:
        """
        Экспоненциальная модель падения плотности с высотой.
        """
        return self.rho0 * np.exp(-h / self.H)