import numpy as np
from scipy.special import wofz as Faddeeva_function
from scipy import integrate


def lambdaDebye(Te, ne):
    """
    Calculates the Debye length in meters as function of electron temperature in eV and
    electron density in 10^22 /m^3
    """
    return 7.43394199e-8 * np.sqrt(np.abs(Te / ne))


def plasma_dispersion_func_deriv(zeta):
    """
    Calculates the plasma dielectric fuction, that is related to the plasma dispersion
    fuction by (-1/2) * derivative of the plasma dispersion function.
    """
    #
    return (1 + zeta * 1j * np.sqrt(np.pi) * Faddeeva_function(zeta))


def invexp(p):
    return np.exp(p**2)


def Rw(x):
    xe = np.array([x]).flatten()
    result_int = np.zeros(len(xe))
    for i in range(len(xe)):
        result_int[i] = integrate.quad(invexp, 0, xe[i], limit=5000)[0]
    return 1 - 2 * xe * np.exp(-xe**2) * result_int


def Iw(x):
    return np.sqrt(np.pi) * x * np.exp(-1. * x**2)


def S_CTS(wavelength, Te, ne, theta=np.pi / 2., Ti=None):
    """"Calculates the Thomson scattering spectrum"""
    Z = 1
    Ti = Ti if Ti is not None else Te
    a = np.sqrt(2 * np.abs(Te) / 510.999e3)
    b = np.sqrt(2 * np.abs(Ti) / 938272e3)
    xe = (wavelength - 532) / ((2 * 532 * np.sin(theta / 2.)) * a)
    xi = (wavelength - 532) / ((2 * 532 * np.sin(theta / 2.)) * b)
    lambdaD = lambdaDebye(Te, ne)
    k = (4 * np.pi / 532e-9) * np.sin(theta / 2.)
    alpha = np.ones(len(wavelength)) / (k * lambdaD)
    we = plasma_dispersion_func_deriv(xe)
    wi = plasma_dispersion_func_deriv(xi)
    Ae = np.exp(-1 * xe**2) * ((1 + alpha**2 * Z * (Te / Ti) * wi.real)**2 + (alpha**2 * Z * (Te / Ti) * wi.imag)**2)
    Ai = Z * np.sqrt(1836.15 * Te / Ti) * np.exp(-1 * xi**2) * ((alpha**2 * we.real)**2 + (alpha**2 * we.imag)**2)
    epsilon2 = (1 + alpha**2 * (we.real + Z * (Te / Ti) * wi.real))**2 + \
               (alpha**2 * we.imag + alpha**2 * Z * (Te / Ti) * wi.imag)**2
    return (2 * np.sqrt(np.pi) / (k * a)) * (Ae / epsilon2 + Ai / epsilon2)


def Se_CTS(wavelength, Te, ne, theta=np.pi / 2., Ti=None):
    """"calculate the Thomson scattering spectrum"""
    Z = 1
    Ti = Ti if Ti is not None else Te
    a = np.sqrt(2 * np.abs(Te) / 510999)
    b = np.sqrt(2 * np.abs(Ti) / 938272e3)
    xe = (wavelength - 532) / ((2 * 532 * np.sin(theta / 2.)) * a)
    xi = (wavelength - 532) / ((2 * 532 * np.sin(theta / 2.)) * b)
    lambdaD = lambdaDebye(Te, ne)
    k = (4 * np.pi / 532e-9) * np.sin(theta / 2.)
    alpha = np.ones(len(wavelength)) / (k * lambdaD)
    we = plasma_dispersion_func_deriv(xe)
    wi = plasma_dispersion_func_deriv(xi)
    Ae = np.exp(-1 * xe**2) * ((1 + alpha**2 * Z * (Te / Ti) * wi.real)**2 + (alpha**2 * Z * (Te / Ti) * wi.imag)**2)
    # Ai = Z * np.sqrt(1836.15 * Te / Ti) * np.exp(-1 * xi**2) * ((alpha**2 * we.real)**2 + (alpha**2 * we.imag)**2)
    epsilon2 = (1 + alpha**2 * (we.real + Z * (Te / Ti) * wi.real))**2 + \
               (alpha**2 * we.imag + alpha**2 * Z * (Te / Ti) * wi.imag)**2
    return (2 * np.sqrt(np.pi) / (k * a)) * (Ae / epsilon2)


def Si_CTS(wavelength, Te, ne, theta=np.pi / 2., Ti=None):
    """"Calculates the Thomson scattering spectrum"""
    Z = 1
    Ti = Ti if Ti is not None else Te
    a = np.sqrt(2 * np.abs(Te) / 510999)
    b = np.sqrt(2 * np.abs(Ti) / 938272e3)
    xe = (wavelength - 532) / ((2 * 532 * np.sin(theta / 2.)) * a)
    xi = (wavelength - 532) / ((2 * 532 * np.sin(theta / 2.)) * b)
    lambdaD = lambdaDebye(Te, ne)
    k = (4 * np.pi / 532e-9) * np.sin(theta / 2.)
    alpha = np.ones(len(wavelength)) / (k * lambdaD)
    we = plasma_dispersion_func_deriv(xe)
    wi = plasma_dispersion_func_deriv(xi)
    # Ae = np.exp(-1 * xe**2) * ((1 + alpha**2 * Z * (Te / Ti) * wi.real)**2 + (alpha**2 * Z * (Te / Ti) * wi.imag)**2)
    Ai = Z * np.sqrt(1836.15 * Te / Ti) * np.exp(-1 * xi**2) * ((alpha**2 * we.real)**2 + (alpha**2 * we.imag)**2)
    epsilon2 = (1 + alpha**2 * (we.real + Z * (Te / Ti) * wi.real))**2 + \
               (alpha**2 * we.imag + alpha**2 * Z * (Te / Ti) * wi.imag)**2
    return (2 * np.sqrt(np.pi) / (k * a)) * (Ai / epsilon2)


def Se_Salpeter(wavelength, Te, ne, theta=np.pi / 2.):
    """"Calculates the electorn feature of the collective thomson scattering spectrum
    with the Sapeter approximation."""
    a = np.sqrt(2 * np.abs(Te) / 510999)
    xe = (wavelength - 532) / ((2 * 532 * np.sin(theta / 2.)) * np.sqrt(2 * np.abs(Te) / 510999))
    lambdaD = lambdaDebye(Te, ne)
    k = (4 * np.pi / 532e-9) * np.sin(theta / 2.)
    alpha = np.ones(len(wavelength)) / (k * lambdaD)
    w = plasma_dispersion_func_deriv(xe)
    return (2 * np.sqrt(np.pi) / (k * a)) * np.exp(-1 * np.square(xe)) / \
           ((1 + alpha**2 * w.real)**2 + (alpha**2 * w.imag) ** 2)


def Si_Salpeter(wavelength, Te, ne, Z=1, theta=np.pi / 2., Ti=None):
    """"Calculates the ion feature of the collective thomson scattering spectrum
    with the Sapeter approximation."""
    Ti = Ti if Ti is not None else Te
    b = np.sqrt(2 * np.abs(Ti) / 938272e3)
    xi = (wavelength - 532) / ((2 * 532 * np.sin(theta / 2.)) * b)
    lambdaD = lambdaDebye(Te, ne)
    k = (4 * np.pi / 532e-9) * np.sin(theta / 2.)
    alpha = np.ones(len(wavelength)) / (k * lambdaD)
    beta = np.sqrt(Z * (alpha**2 / (1 + alpha**2)) * Te / Ti)
    wi = plasma_dispersion_func_deriv(xi)
    return (2 * np.sqrt(np.pi) / (k * b)) * Z * (alpha**2 / (1 + alpha**2))**2 * np.exp(-1 * np.square(xi)) / \
           ((1 + beta**2 * wi.real)**2 + (beta**2 * wi.imag)**2)


def Se(wavelength, Te, ne, theta=np.pi / 2.):
    a = np.sqrt(2 * np.abs(Te) / 510999)
    xe = (wavelength - 532) / ((2 * 532 * np.sin(theta / 2.)) * a)
    lambdaD = lambdaDebye(Te, ne)
    k = (4 * np.pi / 532e-9) * np.sin(theta / 2.)
    alpha = np.ones(len(wavelength)) / (k * lambdaD)
    w = plasma_dispersion_func_deriv(xe)
    return (2 * np.sqrt(np.pi) / (k * a)) * np.exp(-1 * np.square(xe)) / \
           ((1 + alpha**2 * w.real)**2 + (alpha**2 * w.imag) ** 2)


def Si(wavelength, Te, ne, Z=1, theta=np.pi / 2., Ti=None):
    Ti = Ti if Ti is not None else Te
    b = np.sqrt(2 * np.abs(Ti) / 938272e3)
    xi = (wavelength - 532) / ((2 * 532 * np.sin(theta / 2.)) * b)
    lambdaD = lambdaDebye(Te, ne)
    k = (4 * np.pi / 532e-9) * np.sin(theta / 2.)
    alpha = 1 / (k * lambdaD)
    beta = np.sqrt(Z * (alpha**2 / (1 + alpha**2)) * Te / Ti) * np.ones(len(wavelength))
    w = plasma_dispersion_func_deriv(xi)
    return (2 * np.sqrt(np.pi) / (k * b)) * Z * (alpha**2 / (1 + alpha**2))**2 * np.exp(-1 * np.square(xi)) / \
           ((1 + beta**2 * w.real)**2 + (beta**2 * w.imag) ** 2)


def Se_norm(xe, alpha):
    alpha = alpha * np.ones(len(xe))
    w = plasma_dispersion_func_deriv(xe)
    return (1 / np.sqrt(np.pi)) * np.exp(-1 * np.square(xe)) / \
           ((1 + alpha**2 * w.real)**2 + (alpha**2 * w.imag) ** 2)


def Si_norm(xi, alpha, Te_over_Ti=1, Z=1):
    beta = np.sqrt(Z * (alpha**2 / (1 + alpha**2)) * Te_over_Ti) * np.ones(len(xi))
    w = plasma_dispersion_func_deriv(xi)
    return (1 / np.sqrt(np.pi)) * Z * (alpha**2 / (1 + alpha**2))**2 * np.exp(-1 * np.square(xi)) / \
           ((1 + beta**2 * w.real)**2 + (beta**2 * w.imag) ** 2)


if __name__ == '__main__':
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(1.4 * 4.5 * 2, 1.4 * 3))
    wavelength = np.linspace(524, 540, 1000)

    ax1.plot(wavelength, Se_Salpeter(wavelength, 1, 2) / Se_Salpeter(wavelength, 1, 2).max(),
             color='C0', linestyle='solid', label=r"$n_e=2\times10^{22}/m^3$")
    ax1.plot(wavelength, Se_Salpeter(wavelength, 1.5, 2) /
             Se_Salpeter(wavelength, 1.5, 2).max(), color='C0', linestyle='dashed')
    ax1.plot(wavelength, Se_Salpeter(wavelength, 2, 2) /
             Se_Salpeter(wavelength, 2, 2).max(), color='C0', linestyle='dashdot')
    ax1.plot(wavelength, Se_Salpeter(wavelength, 1, 5) / Se_Salpeter(wavelength, 1, 5).max(),
             color='C1', linestyle='solid', label=r"$n_e=5\times10^{22}/m^3$")
    ax1.plot(wavelength, Se_Salpeter(wavelength, 1.5, 5) /
             Se_Salpeter(wavelength, 1.5, 5).max(), color='C1', linestyle='dashed')
    ax1.plot(wavelength, Se_Salpeter(wavelength, 2, 5) /
             Se_Salpeter(wavelength, 2, 5).max(), color='C1', linestyle='dashdot')
    ax1.set_xlim(526, 532)
    ax1.set_ylim(0, 1.1)
    ax1.set_ylabel(r"$Se(\lambda)$")
    ax1.set_xlabel(r"wavelength [nm]")
    ax1.grid()
    ax1.legend(loc='best')
    ax1.text(531.7, 0.22, "1 eV", va='center', ha="center", bbox={
             'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.5})
    ax1.text(531.7, 0.42, "1.5 eV", va='center', ha="center", bbox={
             'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.5})
    ax1.text(527.8, 0.3, "2 eV", va='center', ha="center", bbox={
             'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.5})

    ax1.text(531.7, 0.78, "1 eV", va='center', ha="center", bbox={
             'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.5})
    ax1.text(531.7, 1.04, "2 eV", va='center', ha="center", bbox={
             'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.5})
    ax1.text(531.7, 0.89, "1.5 eV", va='center', ha="center", bbox={
             'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.5})

    ax2.plot(wavelength, Se_Salpeter(wavelength, 1, 10) / Se_Salpeter(wavelength, 1, 10).max(),
             color='C0', linestyle='solid', label=r"$n_e=10\times10^{22}/m^3$")
    ax2.plot(wavelength, Se_Salpeter(wavelength, 1.5, 10) /
             Se_Salpeter(wavelength, 1.5, 10).max(), color='C0', linestyle='dashed')
    ax2.plot(wavelength, Se_Salpeter(wavelength, 2, 10) /
             Se_Salpeter(wavelength, 2, 10).max(), color='C0', linestyle='dashdot')
    ax2.plot(wavelength, Se_Salpeter(wavelength, 1, 15) / Se_Salpeter(wavelength, 1, 15).max(),
             color='C1', linestyle='solid', label=r"$n_e=15\times10^{22}/m^3$")
    ax2.plot(wavelength, Se_Salpeter(wavelength, 1.5, 15) /
             Se_Salpeter(wavelength, 1.5, 15).max(), color='C1', linestyle='dashed')
    ax2.plot(wavelength, Se_Salpeter(wavelength, 2, 15) /
             Se_Salpeter(wavelength, 2, 15).max(), color='C1', linestyle='dashdot')
    ax2.set_xlim(526, 532)
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel(r"$Se(\lambda)$")
    ax2.set_xlabel(r"wavelength [nm]")
    ax2.grid()
    ax2.legend(loc='best')
    ax2.text(526.8, 0.2, "2 eV", va='center', ha="center", bbox={
             'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.5})
    ax2.text(528, 1.05, "1.5 eV", va='center', ha="center", bbox={
             'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.5})
    ax2.text(528.6, 0.1, "1 eV", va='center', ha="center", bbox={
             'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.5})

    ax2.text(531.6, 0.36, "2 eV", va='center', ha="center", bbox={
             'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.5})
    ax2.text(531.6, 0.22, "1.5 eV", va='center', ha="center", bbox={
             'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.5})
    ax2.text(529, 0.9, "1 eV", va='center', ha="center", bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.5})

    # ax2.text(0.05, 0.95, "b) $T_e=2$ eV", transform=ax2.transAxes,
    #          verticalalignment='top', bbox={'boxstyle': 'round', 'facecolor': 'white'})
    fig.tight_layout()
    # fig.savefig("Salpetier_e.png")
    plt.show()

    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(1.4 * 4.5 * 2, 1.4 * 3))
    wavelength = np.linspace(532-0.2, 532+0.2, 1000)

    ax1.plot(wavelength, Si_Salpeter(wavelength, 1, 2) / Si_Salpeter(wavelength, 1, 2).max(),
             color='C0', linestyle='solid', label=r"$T_e=1eV; n_e=2\times10^{22}/m^3$")
    # ax1.plot(wavelength, Si_Salpeter(wavelength, 1.5, 2) / Si_Salpeter(wavelength, 1.5, 2).max(), 
    #         color='C0', linestyle='dashed')
    ax1.plot(wavelength, Si_Salpeter(wavelength, 2, 2) /  Si_Salpeter(wavelength, 2, 2).max(), 
        color='C0', linestyle='dashdot', label=r"$T_e=2eV; n_e=2\times10^{22}/m^3$")
    ax1.plot(wavelength, Si_Salpeter(wavelength, 1, 5) / Si_Salpeter(wavelength, 1, 5).max(),
             color='C1', linestyle='solid', label=r"$T_e=2eV; n_e=5\times10^{22}/m^3$")
    # ax1.plot(wavelength, Si_Salpeter(wavelength, 1.5, 5) / Si_Salpeter(wavelength, 1.5, 5).max(), color='C1', linestyle='dashed')
    ax1.plot(wavelength, Si_Salpeter(wavelength, 2, 5) / Si_Salpeter(wavelength, 2, 5).max(), color='C1', linestyle='dashdot', label=r"$T_e=1eV; n_e=5\times10^{22}/m^3$")
    ax1.set_xlim(532-0.15, 532+0.15)
    ax1.set_ylim(0, 1.1)
    ax1.set_ylabel(r"$Se(\lambda)$ [arb. un.]")
    ax1.set_xlabel(r"wavelength [nm]")
    ax1.grid()
    ax1.legend(loc='best')

    ax2.plot(wavelength, Si_Salpeter(wavelength, 1, 10) / Si_Salpeter(wavelength, 1, 10).max(),
             color='C0', linestyle='solid', label=r"$T_e=1eV; n_e=10\times1^{23}/m^3$")
    # ax2.plot(wavelength, Si_Salpeter(wavelength, 1.5, 10) /
    #          Si_Salpeter(wavelength, 1.5, 10).max(), color='C0', linestyle='dashed')
    ax2.plot(wavelength, Si_Salpeter(wavelength, 2, 10) /
             Si_Salpeter(wavelength, 2, 10).max(), color='C0', linestyle='dashdot', label=r"$T_e=2eV; n_e=1\times10^{23}/m^3$")
    ax2.plot(wavelength, Si_Salpeter(wavelength, 1, 15) / Si_Salpeter(wavelength, 1, 15).max(),
             color='C1', linestyle='solid', label=r"$T_e=1eV; n_e=1.5\times10^{23}/m^3$")
    # ax2.plot(wavelength, Si_Salpeter(wavelength, 1.5, 15) /
    #          Si_Salpeter(wavelength, 1.5, 15).max(), color='C1', linestyle='dashed')
    ax2.plot(wavelength, Si_Salpeter(wavelength, 2, 15) /
             Si_Salpeter(wavelength, 2, 15).max(), color='C1', linestyle='dashdot', label=r"$T_e=2eV; n_e=15\times1.5^{22}/m^3$")
    ax2.set_xlim(532-0.15, 532+0.15)
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel(r"$Se(\lambda)$ [arb. un.]" )
    ax2.set_xlabel(r"wavelength [nm]")
    ax2.grid()
    ax2.legend(loc='best')
    # ax2.text(0.05, 0.95, "b) $T_e=2$ eV", transform=ax2.transAxes,
    #          verticalalignment='top', bbox={'boxstyle': 'round', 'facecolor': 'white'})
    fig.tight_layout()
    # fig.savefig("Salpetier_i.png")
    plt.show()

    fig, ax = plt.subplots(figsize=(1.4 * 4.5, 1.4 * 3))
    x = np.linspace(0, 4, 200)
    ax.plot(x, plasma_dispersion_func_deriv(x).real, label="Rw(x)")
    ax.plot(x, plasma_dispersion_func_deriv(x).imag, '--', label="Iw(x)")
    ax.set_xlabel("x")
    ax.grid()
    ax.legend(loc=1)
    ax.set_xlim(0, 4)
    ax.set_ylim(-0.3, 1)
    fig.tight_layout()
    # fig.savefig("Rw_Iw.png")
    plt.show()

    fig, ax = plt.subplots(figsize=(1.4 * 4.5, 1.4 * 3))
    x = np.linspace(-4, 4, 200)
    ax.plot(x, Rw(x) - plasma_dispersion_func_deriv(x).real, label="diff Rw(x)")
    ax.plot(x, Iw(x) - plasma_dispersion_func_deriv(x).imag, label="diff Iw(x)")
    ax.set_xlabel("x")
    ax.grid()
    ax.legend()
    ax.set_xlim(-4, 4)
    fig.tight_layout()
    plt.show()

    x = np.linspace(-3, 3, 500)
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(1.4 * 4.5 * 2, 1.4 * 3))
    ax1.plot(x, Se_norm(x, 0), label=r"$\alpha=0$")
    ax1.plot(x, Se_norm(x, 0.5), label=r"$\alpha=0.5$")
    ax1.plot(x, Se_norm(x, 1), label=r"$\alpha=1$")
    ax1.plot(x, Se_norm(x, 2), label=r"$\alpha=2$")
    ax1.plot(x, Se_norm(x, 3), label=r"$\alpha=3$")
    ax1.set_ylabel("$S_e$ [arb. un.]")
    ax1.set_xlabel("$x_e$")
    ax1.grid()
    ax1.legend()
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(0, 0.6)

    ax2.plot(x, Si_norm(x, 0.5), label=r"$\alpha=0.5$")
    ax2.plot(x, Si_norm(x, 1), label=r"$\alpha=1$")
    ax2.plot(x, Si_norm(x, 2), label=r"$\alpha=2$")
    ax2.plot(x, Si_norm(x, 3), label=r"$\alpha=3$")
    ax2.set_ylabel("$S_i$ [arb. un.]")
    ax2.set_xlabel("$x_i$")
    ax2.grid()
    ax2.legend()
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(0, 0.12)
    fig.tight_layout()
    # fig.savefig("Se_Si.png")
    plt.show()

    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(1.4 * 4.5 * 2, 1.4 * 3))
    wavelength = np.linspace(532 - 5, 532 + 5, 2000)
    ax1.plot(wavelength, Se(wavelength, 1, 0.5) + Si(wavelength, 1, 0.5), label=r"$n_e=0.5\times10^{22}/m^3$")
    ax1.plot(wavelength, Se(wavelength, 1, 1) + Si(wavelength, 1, 1), label=r"$n_e=1\times10^{22}/m^3$")
    ax1.plot(wavelength, Se(wavelength, 1, 2) + Si(wavelength, 1, 2), label=r"$n_e=2\times10^{22}/m^3$")
    ax1.plot(wavelength, Se(wavelength, 1, 5) + Si(wavelength, 1, 5), label=r"$n_e=5\times10^{22}/m^3$")
    ax1.plot(wavelength, Se(wavelength, 1, 10) + Si(wavelength, 1, 10), label=r"$n_e=10\times10^{22}/m^3$")
    ax1.set_xlim(wavelength.min(), wavelength.max())
    ax1.set_yscale('log')
    ax1.set_ylabel(r"$S(\lambda)$")
    ax1.set_xlabel(r"wavelength [nm]")
    ax1.grid()
    ax1.legend()
    ax1.text(0.05, 0.95, "a) $T_e=1$ eV", transform=ax1.transAxes,
             verticalalignment='top', bbox={'boxstyle': 'round', 'facecolor': 'white'})

    ax2.plot(wavelength, Se(wavelength, 2, 0.5) + Si(wavelength, 2, 0.5), label=r"$n_e=0.5\times10^{22}/m^3$")
    ax2.plot(wavelength, Se(wavelength, 2, 1) + Si(wavelength, 2, 1), label=r"$n_e=1\times10^{22}/m^3$")
    ax2.plot(wavelength, Se(wavelength, 2, 2) + Si(wavelength, 2, 2), label=r"$n_e=2\times10^{22}/m^3$")
    ax2.plot(wavelength, Se(wavelength, 2, 5) + Si(wavelength, 2, 5), label=r"$n_e=5\times10^{22}/m^3$")
    ax2.plot(wavelength, Se(wavelength, 2, 10) + Si(wavelength, 2, 10), label=r"$n_e=10\times10^{22}/m^3$")
    ax2.set_xlim(wavelength.min(), wavelength.max())
    ax2.set_yscale('log')
    ax2.set_ylabel(r"$S(\lambda)$")
    ax2.set_xlabel(r"wavelength [nm]")
    ax2.grid()
    ax2.legend()
    ax2.text(0.05, 0.95, "b) $T_e=2$ eV", transform=ax2.transAxes,
             verticalalignment='top', bbox={'boxstyle': 'round', 'facecolor': 'white'})
    fig.tight_layout()
    # fig.savefig("Se_Si_532nm.png")
    plt.show()

    fig = plt.figure(figsize=(1.4 * 4.5 * 2, 1.4 * 3))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    wavelength = np.linspace(532 - 5, 532 + 5, 2000)
    y = np.ones(wavelength.size)
    ne_list = np.linspace(1, 10, 10)
    for ne in ne_list:
        ax1.plot(wavelength, ne * np.ones(wavelength.size),
                 np.log(S_CTS(wavelength, 1, ne)), color='#1f77b4')

    ax1.set_xlim(wavelength.min(), wavelength.max())
    ax1.set_ylim(ne_list.max(), 0)
    ax1.set_zlabel(r"$\log$  S($\lambda$)")
    ax1.set_xlabel(r"wavelength [nm]")
    ax1.set_ylabel(r"$n_e$ [$10^{22} m^{-3}]$")
    ax1.grid()
    ax1.text2D(0.05, 0.95, "a) $T_e=1$ eV", transform=ax1.transAxes,
               verticalalignment='top', bbox={'boxstyle': 'round', 'facecolor': 'white'})

    for ne in ne_list:
        ax2.plot(wavelength, ne * np.ones(wavelength.size),
                 np.log(S_CTS(wavelength, 2, ne)), color='#1f77b4')

    ax2.set_xlim(wavelength.min(), wavelength.max())
    ax2.set_ylim(ne_list.max(), 0)
    ax2.set_zlabel(r"$\log$ S($\lambda$)")
    ax2.set_xlabel(r"wavelength [nm]")
    ax2.set_ylabel(r"$n_e$ [$10^{22} m^{-3}]$")
    ax2.grid()
    ax2.text2D(0.05, 0.95, "b) $T_e=2$ eV", transform=ax2.transAxes,
               verticalalignment='top', bbox={'boxstyle': 'round', 'facecolor': 'white'})
    fig.tight_layout()
    # fig.savefig("Se_Si_532nm_3.png")
    plt.show()

    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(1.4 * 4.5 * 2, 1.4 * 3))
    fig.gca().set_prop_cycle(None)
    wavelength = np.linspace(532 - 5, 532 + 5, 5000)
    ax1.plot(wavelength, S_CTS(wavelength, 1, 0.5), label=r"$n_e=0.5\times10^{22}/m^3$")
    ax1.plot(wavelength, S_CTS(wavelength, 1, 1), label=r"$n_e=1\times10^{22}/m^3$")
    ax1.plot(wavelength, S_CTS(wavelength, 1, 2), label=r"$n_e=2\times10^{22}/m^3$")
    ax1.plot(wavelength, S_CTS(wavelength, 1, 5), label=r"$n_e=5\times10^{22}/m^3$")
    ax1.plot(wavelength, S_CTS(wavelength, 1, 10), label=r"$n_e=10\times10^{22}/m^3$")
    fig.gca().set_prop_cycle(None)
    ax1.plot(wavelength, Se_Salpeter(wavelength, 1, 0.5) + Si_Salpeter(wavelength, 1, 0.5), '--')
    ax1.plot(wavelength, Se_Salpeter(wavelength, 1, 1) + Si_Salpeter(wavelength, 1, 1), '--')
    ax1.plot(wavelength, Se_Salpeter(wavelength, 1, 2) + Si_Salpeter(wavelength, 1, 2), '--')
    ax1.plot(wavelength, Se_Salpeter(wavelength, 1, 5) + Si_Salpeter(wavelength, 1, 5), '--')
    ax1.plot(wavelength, Se_Salpeter(wavelength, 1, 10) + Si_Salpeter(wavelength, 1, 10), '--')
    ax1.set_xlim(wavelength.min(), wavelength.max())
    ax1.set_ylim(1e-8, 2e-3)
    ax1.set_yscale('log')
    ax1.set_ylabel(r"$S(\lambda)$")
    ax1.set_xlabel(r"wavelength [nm]")
    ax1.grid()
    ax1.legend(loc=1)
    ax1.text(0.05, 0.95, "a) $T_e=1$ eV", transform=ax1.transAxes,
             verticalalignment='top', bbox={'boxstyle': 'round', 'facecolor': 'white'})

    ax2.plot(wavelength, S_CTS(wavelength, 2, 0.5), label=r"$n_e=0.5\times10^{22}/m^3$")
    ax2.plot(wavelength, S_CTS(wavelength, 2, 1), label=r"$n_e=1\times10^{22}/m^3$")
    ax2.plot(wavelength, S_CTS(wavelength, 2, 2), label=r"$n_e=2\times10^{22}/m^3$")
    ax2.plot(wavelength, S_CTS(wavelength, 2, 5), label=r"$n_e=5\times10^{22}/m^3$")
    ax2.plot(wavelength, S_CTS(wavelength, 2, 10), label=r"$n_e=10\times10^{22}/m^3$")
    fig.gca().set_prop_cycle(None)
    ax2.plot(wavelength, Se_Salpeter(wavelength, 2, 0.5) + Si_Salpeter(wavelength, 2, 0.5), '--')
    ax2.plot(wavelength, Se_Salpeter(wavelength, 2, 1) + Si_Salpeter(wavelength, 2, 1), '--')
    ax2.plot(wavelength, Se_Salpeter(wavelength, 2, 2) + Si_Salpeter(wavelength, 2, 2), '--')
    ax2.plot(wavelength, Se_Salpeter(wavelength, 2, 5) + Si_Salpeter(wavelength, 2, 5), '--')
    ax2.plot(wavelength, Se_Salpeter(wavelength, 2, 10) + Si_Salpeter(wavelength, 2, 10), '--')
    ax2.set_xlim(wavelength.min(), wavelength.max())
    ax2.set_ylim(1e-6, 1e-3)
    ax2.set_yscale('log')
    ax2.set_ylabel(r"$S(\lambda)$")
    ax2.set_xlabel(r"wavelength [nm]")
    ax2.grid()
    ax2.legend(loc=1)
    ax2.text(0.05, 0.95, "b) $T_e=2$ eV", transform=ax2.transAxes,
             verticalalignment='top', bbox={'boxstyle': 'round', 'facecolor': 'white'})
    fig.tight_layout()
    # fig.savefig("CTS_vs_Salpetier_e+i.png")
    plt.show()

    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(1.4 * 4.5 * 2, 1.4 * 3))
    wavelength = np.linspace(532 - 0.10, 532 + 0.10, 500)
    ax1.plot(wavelength, S_CTS(wavelength, 1, 0.5) -
             Se_Salpeter(wavelength, 1, 0.5), label=r"$n_e=0.5\times10^{22}/m^3$")
    ax1.plot(wavelength, S_CTS(wavelength, 1, 1) - Se_Salpeter(wavelength, 1, 1), label=r"$n_e=1\times10^{22}/m^3$")
    ax1.plot(wavelength, S_CTS(wavelength, 1, 5) - Se_Salpeter(wavelength, 1, 5), label=r"$n_e=5\times10^{22}/m^3$")
    ax1.plot(wavelength, S_CTS(wavelength, 1, 10) - Se_Salpeter(wavelength, 1, 10), label=r"$n_e=10\times10^{22}/m^3$")
    ax1.set_prop_cycle(None)
    ax1.plot(wavelength, Si_Salpeter(wavelength, 1, 0.5), '--')
    ax1.plot(wavelength, Si_Salpeter(wavelength, 1, 1), '--')
    ax1.plot(wavelength, Si_Salpeter(wavelength, 1, 5), '--')
    ax1.plot(wavelength, Si_Salpeter(wavelength, 1, 10), '--')
    ax1.set_xlim(wavelength.min(), wavelength.max())
    ax1.set_ylabel(r"$S(\lambda)$")
    ax1.set_xlabel(r"wavelength [nm]")
    ax1.grid()
    ax1.legend(loc=1)
    ax1.text(0.05, 0.95, "a) $T_e=1$ eV", transform=ax1.transAxes,
             verticalalignment='top', bbox={'boxstyle': 'round', 'facecolor': 'white'})

    ax2.plot(wavelength, S_CTS(wavelength, 2, 0.5) -
             Se_Salpeter(wavelength, 2, 0.5), label=r"$n_e=0.5\times10^{22}/m^3$")
    ax2.plot(wavelength, S_CTS(wavelength, 2, 10) - Se_Salpeter(wavelength, 2, 10), label=r"$n_e=10\times10^{22}/m^3$")
    ax2.set_prop_cycle(None)
    ax2.plot(wavelength, Si_Salpeter(wavelength, 2, 0.5), '--')
    ax2.plot(wavelength, Si_Salpeter(wavelength, 2, 10), '--')
    ax2.set_xlim(wavelength.min(), wavelength.max())
    ax2.set_ylabel(r"$S(\lambda)$")
    ax2.set_xlabel(r"wavelength [nm]")
    ax2.grid()
    ax2.legend(loc=1)
    ax2.text(0.05, 0.95, "b) $T_e=2$ eV", transform=ax2.transAxes,
             verticalalignment='top', bbox={'boxstyle': 'round', 'facecolor': 'white'})
    fig.tight_layout()
    # fig.savefig("ion_feature_full_Salpeter.png")
    plt.show()

    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(1.4 * 4.5 * 2, 1.4 * 3))
    wavelength = np.linspace(532 - 0.10, 532 + 0.10, 500)
    ax1.plot(wavelength, Se_CTS(wavelength, 1, 0.5), label=r"$n_e=0.5\times10^{22}/m^3$")
    ax1.plot(wavelength, Se_CTS(wavelength, 1, 1), label=r"$n_e=1\times10^{22}/m^3$")
    ax1.plot(wavelength, Se_CTS(wavelength, 1, 5), label=r"$n_e=5\times10^{22}/m^3$")
    ax1.plot(wavelength, Se_CTS(wavelength, 1, 10), label=r"$n_e=10\times10^{22}/m^3$")
    ax1.set_prop_cycle(None)
    ax1.plot(wavelength, Se_Salpeter(wavelength, 1, 0.5), '--')
    ax1.plot(wavelength, Se_Salpeter(wavelength, 1, 1), '--')
    ax1.plot(wavelength, Se_Salpeter(wavelength, 1, 5), '--')
    ax1.plot(wavelength, Se_Salpeter(wavelength, 1, 10), '--')
    ax1.set_xlim(wavelength.min(), wavelength.max())
    ax1.set_ylabel(r"$S(\lambda)$")
    ax1.set_xlabel(r"wavelength [nm]")
    ax1.grid()
    ax1.legend(loc=1)
    ax1.text(0.05, 0.95, "a) $T_e=1$ eV", transform=ax1.transAxes,
             verticalalignment='top', bbox={'boxstyle': 'round', 'facecolor': 'white'})

    ax2.plot(wavelength, Se_CTS(wavelength, 2, 0.5), label=r"$n_e=0.5\times10^{22}/m^3$")
    ax2.plot(wavelength, Se_CTS(wavelength, 2, 1), label=r"$n_e=1\times10^{22}/m^3$")
    ax2.plot(wavelength, Se_CTS(wavelength, 2, 5), label=r"$n_e=5\times10^{22}/m^3$")
    ax2.plot(wavelength, Se_CTS(wavelength, 2, 10), label=r"$n_e=10\times10^{22}/m^3$")
    ax2.set_prop_cycle(None)
    ax2.plot(wavelength, Se_Salpeter(wavelength, 2, 0.5), '--')
    ax2.plot(wavelength, Se_Salpeter(wavelength, 2, 1), '--')
    ax2.plot(wavelength, Se_Salpeter(wavelength, 2, 5), '--')
    ax2.plot(wavelength, Se_Salpeter(wavelength, 2, 10), '--')
    ax2.set_xlim(wavelength.min(), wavelength.max())
    ax2.set_ylabel(r"$S(\lambda)$")
    ax2.set_xlabel(r"wavelength [nm]")
    ax2.grid()
    ax2.legend(loc=1)
    ax2.text(0.05, 0.95, "b) $T_e=2$ eV", transform=ax2.transAxes,
             verticalalignment='top', bbox={'boxstyle': 'round', 'facecolor': 'white'})
    fig.tight_layout()
    # fig.savefig("electron_feature_vs_Salpeter.png")
    plt.show()

    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(1.4 * 4.5 * 2, 1.4 * 3))
    wavelength = np.linspace(532 - 0.10, 532 + 0.10, 500)
    ax1.plot(wavelength, Si_CTS(wavelength, 1, 0.5), label=r"$n_e=0.5\times10^{22}/m^3$")
    ax1.plot(wavelength, Si_CTS(wavelength, 1, 1), label=r"$n_e=1\times10^{22}/m^3$")
    ax1.plot(wavelength, Si_CTS(wavelength, 1, 5), label=r"$n_e=5\times10^{22}/m^3$")
    ax1.plot(wavelength, Si_CTS(wavelength, 1, 10), label=r"$n_e=10\times10^{22}/m^3$")
    ax1.set_prop_cycle(None)
    ax1.plot(wavelength, Si_Salpeter(wavelength, 1, 0.5), '--')
    ax1.plot(wavelength, Si_Salpeter(wavelength, 1, 1), '--')
    ax1.plot(wavelength, Si_Salpeter(wavelength, 1, 5), '--')
    ax1.plot(wavelength, Si_Salpeter(wavelength, 1, 10), '--')
    ax1.set_xlim(wavelength.min(), wavelength.max())
    ax1.set_ylabel(r"$S(\lambda)$")
    ax1.set_xlabel(r"wavelength [nm]")
    ax1.grid()
    ax1.legend(loc=1)
    ax1.text(0.05, 0.95, "a) $T_e=1$ eV", transform=ax1.transAxes,
             verticalalignment='top', bbox={'boxstyle': 'round', 'facecolor': 'white'})

    ax2.plot(wavelength, Si_CTS(wavelength, 2, 0.5), label=r"$n_e=0.5\times10^{22}/m^3$")
    ax2.plot(wavelength, Si_CTS(wavelength, 2, 1), label=r"$n_e=1\times10^{22}/m^3$")
    ax2.plot(wavelength, Si_CTS(wavelength, 2, 5), label=r"$n_e=5\times10^{22}/m^3$")
    ax2.plot(wavelength, Si_CTS(wavelength, 2, 10), label=r"$n_e=10\times10^{22}/m^3$")
    ax2.set_prop_cycle(None)
    ax2.plot(wavelength, Si_Salpeter(wavelength, 2, 0.5), '--')
    ax2.plot(wavelength, Si_Salpeter(wavelength, 2, 1), '--')
    ax2.plot(wavelength, Si_Salpeter(wavelength, 2, 5), '--')
    ax2.plot(wavelength, Si_Salpeter(wavelength, 2, 10), '--')
    ax2.set_xlim(wavelength.min(), wavelength.max())
    ax2.set_ylabel(r"$S(\lambda)$")
    ax2.set_xlabel(r"wavelength [nm]")
    ax2.grid()
    ax2.legend(loc=1)
    ax2.text(0.05, 0.95, "b) $T_e=2$ eV", transform=ax2.transAxes,
             verticalalignment='top', bbox={'boxstyle': 'round', 'facecolor': 'white'})
    fig.tight_layout()
    # fig.savefig("ion_feature_vs_Salpeter.png")
    plt.show()

    from scipy.constants import epsilon_0, speed_of_light, Planck, electron_mass, elementary_charge, Boltzmann


    def delta_E_E(Te, ne, E_L):
        return 2 * kappa_BI(Te, ne) * E_L / (3 * Boltzmann * Te * ne * np.pi * 0.25e-3**2)


    def kappa_BI(Te, ne, Z=1):
        kappa = np.sqrt(32 * np.pi / 27) * (ne**2 * Z**2 * (532e-9)**3 / (Planck * electron_mass**2 * speed_of_light**4)) * (elementary_charge**2 / (4 * np.pi *
                                                                                                                                           epsilon_0)) ** 3 * np.sqrt(electron_mass / (Boltzmann * Te)) * (1. - np.exp(-1. * Planck * speed_of_light / (Boltzmann * Te * 532e-9))) * 1.2
    #    kappa = np.sqrt(32 * np.pi / 27) * (Z**2 / (electron_mass**2 * speed_of_light**2)) * (elementary_charge**2 / (4 * np.pi * epsilon_0)) ** 3 * (532e-9**3/(Planck)) * np.sqrt(electron_mass / (Boltzmann * Te)) * ne**2 * (1. - np.exp(-1. * Planck * speed_of_light / (Boltzmann * Te * 532e-9))) * 1.2
        return kappa

    def delta_E_E2(Te, ne, E_L):
        return 6.6e-5 * (Z**2 * ne / np.pow(Te, 3./2) * (E_L / (np.pi * 0.25e-3**2)) * 1.2 * 532e-9**3 * (1-np/exp( -1. * Planck * speed_of_light / (Boltzmann * Te * 532e-9))))

    Te = np.linspace(1e3, 25e3, 200)
    delta = delta_E_E(Te, 2e23, 25e-3)
    plt.plot(Te, delta)
    plt.show()
