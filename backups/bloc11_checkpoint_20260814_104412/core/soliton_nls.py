import numpy as np

class SolitonAutonome:
    def __init__(self, L=40.0, N=128, beta=1.0, gamma=5e-5, v0=0.5):
        self.L, self.N = L, N
        self.beta, self.gamma, self.v0 = beta, gamma, v0
        self.x = np.linspace(-L/2, L/2, N, endpoint=False)
        self.dx = self.x[1] - self.x[0]
        self.k = 2 * np.pi * np.fft.fftfreq(N, d=self.dx)
        self.psi0 = (1.0 / np.cosh(self.x)) * np.exp(1j * v0 * self.x)

    def _rhs(self, psi):
        psi_hat = np.fft.fft(psi)
        d2psi = np.fft.ifft(-self.k**2 * psi_hat)
        return 1j * (0.5 * d2psi + self.beta * np.abs(psi)**2 * psi) - self.gamma * psi

    def evoluer(self, t_final=5.0, n_points=50):
        internal_steps = 20
        total_steps = n_points * internal_steps
        dt = t_final / total_steps
        
        psi = self.psi0.copy()
        psi_t = np.zeros((self.N, n_points), dtype=complex)
        psi_t[:, 0] = psi
        
        t_idx = 1
        for step in range(1, total_steps + 1):
            k1 = self._rhs(psi)
            k2 = self._rhs(psi + 0.5 * dt * k1)
            k3 = self._rhs(psi + 0.5 * dt * k2)
            k4 = self._rhs(psi + dt * k3)
            psi = psi + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
            if step % internal_steps == 0 and t_idx < n_points:
                psi_t[:, t_idx] = psi
                t_idx += 1
                
        t_eval = np.linspace(0, t_final, n_points)
        return t_eval, psi_t

    def energie(self, psi):
        psi_hat = np.fft.fft(psi)
        dpsi = np.fft.ifft(1j * self.k * psi_hat)
        E_kin = 0.5 * np.sum(np.abs(dpsi)**2) * self.dx
        E_nl = -0.5 * self.beta * np.sum(np.abs(psi)**4) * self.dx
        return float(E_kin + E_nl)

    def centre_et_largeur(self, psi):
        densite = np.abs(psi)**2
        norme = np.sum(densite) * self.dx + 1e-30
        centre = np.sum(self.x * densite) * self.dx / norme
        largeur = np.sqrt(np.sum((self.x - centre)**2 * densite) * self.dx / norme)
        return float(centre), float(largeur)

    def diagnostic(self, t_final=5.0):
        t, psi_t = self.evoluer(t_final)
        energies = np.array([self.energie(psi_t[:, i]) for i in range(psi_t.shape[1])])
        centres, largeurs = [], []
        for i in range(psi_t.shape[1]):
            c, l = self.centre_et_largeur(psi_t[:, i])
            centres.append(c)
            largeurs.append(l)
        centres, largeurs = np.array(centres), np.array(largeurs)

        energie_init = energies[0]
        derive_energie = float(np.max(np.abs(energies - energie_init)) / (abs(energie_init) + 1e-12))
        derive_largeur = float(np.max(np.abs(largeurs - largeurs[0])) / (largeurs[0] + 1e-12))
        vitesse_obs = float((centres[-1] - centres[0]) / (t[-1] - t[0])) if len(t) > 1 else 0.0
        stable = bool(derive_energie < 0.03 and derive_largeur < 0.15)

        return {
            "energie_initiale": float(energie_init),
            "energie_finale": float(energies[-1]),
            "derive_energie_relative": derive_energie,
            "derive_largeur_relative": derive_largeur,
            "vitesse_imposee": float(self.v0),
            "vitesse_observee": vitesse_obs,
            "stable": stable
        }
