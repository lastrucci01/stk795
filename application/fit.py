import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.mixture import GaussianMixture

np.random.seed(42)


def generate_normal_data(mean=25, std_dev=5, size=1000):
    data = stats.norm.rvs(loc=mean, scale=std_dev, size=size)
    return data


def generate_contaminated_normal(mean=25, std_dev=5, alpha=0.2, lambda_=3, size=1000):

    random_probs = np.random.rand(size)
    normal_component = stats.norm.rvs(loc=mean, scale=std_dev, size=size)
    contaminated_component = stats.norm.rvs(
        loc=mean, scale=lambda_ * std_dev, size=size
    )
    data = np.where(random_probs < alpha, contaminated_component, normal_component)

    return data


def generate_cauchy_data(location=25, scale=5, size=1000):
    data = stats.cauchy.rvs(loc=location, scale=scale, size=size)
    lower_threshold = np.median(data) - 50 * scale
    upper_threshold = np.median(data) + 50 * scale

    data = data[(data > lower_threshold) & (data < upper_threshold)]

    return data


def generate_levy(alpha=1.5, beta=0.0, scale=10, location=0, size=1000):
    data = stats.levy_stable.rvs(alpha, beta, loc=location, scale=scale, size=size)
    lower_threshold = np.median(data) - 50 * scale
    upper_threshold = np.median(data) + 50 * scale

    data = data[(data > lower_threshold) & (data < upper_threshold)]

    return data


def generate_weibull(shape=1.5, scale=25, size=1000):

    data = stats.weibull_min.rvs(shape, scale=scale, size=size)
    return data


def plot_histogram(data, bins=50):

    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, density=True, alpha=0.6, color="#000000")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.grid()
    plt.show()


def AIC(num_params, log_likelihood):
    return 2 * num_params - 2 * log_likelihood

def BIC(num_params, log_likelihood, num_samples):
    return num_params * np.log(num_samples) - 2 * log_likelihood

def fit_norm(data):
    size = len(data)
    mu, std = stats.norm.fit(data)

    log_likelihood = np.sum(np.log(stats.norm.pdf(data, mu, std)))
    num_params = 2
    AIC_norm = AIC(num_params, log_likelihood)
    BIC_norm = BIC(num_params, log_likelihood, size)
    params = {"mu": mu, "std": std}

    return AIC_norm, BIC_norm, params


def plot_norm(data, mu, std, ax=None):
    if ax is None:
        ax = plt.gca()

    ax.hist(
        data, bins=30, density=True, alpha=0.6, color="#000000", label="Data Histogram"
    )

    x = np.linspace(min(data), max(data), 1000)

    pdf_fitted_normal = stats.norm.pdf(x, mu, std)
    ax.plot(
        x, pdf_fitted_normal, color="#d02b35", lw=2, label="Fitted Normal Distribution"
    )

    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid()

def fit_t(data):
    size = len(data)
    df, loc, scale = stats.t.fit(data)
    log_likelihood = np.sum(np.log(stats.t.pdf(data, df, loc, scale)))
    num_params = 3
    AIC_t = AIC(num_params, log_likelihood)
    BIC_t = BIC(num_params, log_likelihood, size)
    params = {"df": df, "loc": loc, "scale": scale}

    return AIC_t, BIC_t, params


def plot_t(data, df, loc, scale, ax=None):
    if ax is None:
        ax = plt.gca()  

    ax.hist(
        data, bins=30, density=True, alpha=0.6, color="#000000", label="Data Histogram"
    )
    x = np.linspace(min(data), max(data), 1000)
    pdf_fitted_t = stats.t.pdf(x, df, loc, scale)
    ax.plot(x, pdf_fitted_t, color="#d02b35", lw=2, label="Fitted t-Distribution")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid()



def fit_gmm(data, n_components=3):
    data_reshaped = data.reshape(-1, 1)

    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(data_reshaped)

    AIC_gmm = gmm.aic(data_reshaped) 
    BIC_gmm = gmm.bic(data_reshaped) 

    return AIC_gmm, BIC_gmm,gmm

def plot_gmm(data, gmm, n_components=3, ax=None):
    if ax is None:
        ax = plt.gca()

    ax.hist(
        data, bins=30, density=True, alpha=0.6, color="#000000", label="Data Histogram"
    )

    x = np.linspace(min(data), max(data), 1000).reshape(-1, 1)
    pdf_fitted_gmm = np.zeros_like(x)

    for i in range(n_components):
        mean = gmm.means_[i, 0]
        variance = gmm.covariances_[i, 0, 0]
        weight = gmm.weights_[i]

        pdf_component = weight * stats.norm.pdf(x, mean, np.sqrt(variance))
        pdf_fitted_gmm += pdf_component
        ax.plot(x, pdf_component, linestyle="--", label=f"Component {i+1}")

    ax.plot(x, pdf_fitted_gmm, color="#d02b35", lw=2, label=f"Fitted GMM (n_components={n_components})")

    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid()