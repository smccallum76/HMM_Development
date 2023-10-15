"""
First pass at building a Hidden Markov Model using the simple simulations data. This code will use portions of the
code from Stat Modeling Final.  The objective of this code is to identify the neutral and sweep events with min
amount of errors
"""
import pandas as pd
import numpy as np
import scipy.stats as stats

def hmm_get_data(path):
    """
    Import data with the expectation of two distributions
    :param path:
    path where the data is stored
    :return:
    Data as a pandas dataframe
    """
    # x = pd.read_csv(path)

    # delete the tranforms below, this was for testing with known data set that had weird formatting. Then
    # uncomment the x statement above
    x = pd.read_csv(path, delimiter=',', header=None)
    x = x.iloc[:, 0:4]
    x = np.array(x)
    x = x.flatten(order='C')
    x = x.reshape((len(x), 1))
    return x
def hmm_intialize(mu_list, sd_list, pi_list):
    """
    Initialization parameters for HMM
    :param mu_list:
    List of mean values for each normal distribution
    :param sd_list:
    List of the standard deviations for each normal distribution
    :param pi_list:
    List of the pi fractions for each class (percentage of each class in data set)
    :return:
    Dataframe containing the mu, sd, and pi values provided
    """
    init = pd.DataFrame(columns=['mu', 'sd', 'pi'])
    init['mu'] = mu_list
    init['sd'] = sd_list
    init['pi'] = pi_list
    return init

def hmm_transition(a_list):
    """
    Reshape a list of transition values into a square matrix
    :param a_list:
    List of transition values.
    Position 0 is transition from 0 to 0
    Position 1 is transition from 0 to 1
    Position 3 is transition from 1 to 0
    Position 4 is transition from 1 to 1...
    :return:
    The square matrix of transition values (A).
    """
    shape = (int(np.sqrt(len(a_list))), int(np.sqrt(len(a_list))))
    a = np.array(a_list)
    a = np.reshape(a, newshape=shape, order='C')
    return a

def hmm_norm_pdf(x, mu, sd):
    """
    Calculate the probability density based on a normal distribution
    :param x:
    Value to be evaluated
    :param mu:
    Mean of the distribution
    :param sd:
    Standard deviation of the distribution
    :return:
    Probability density of the value x
    """
    p = stats.norm.pdf(x=x,  loc=mu, scale=sd)
    return p
def hmm_forward(init, data, A_trans):
    # some initializations and settings
    n = len(data)
    A_new = A_trans

    # Probability density values for the data using distribution 1
    bx1 = hmm_norm_pdf(x=data, mu=init.loc[0, 'mu'], sd=init.loc[0, 'sd'])
    alpha1 = np.array(np.log(bx1[0] * init.loc[0, 'pi']))

    # Probability density values for the data using distribution 2
    bx2 = hmm_norm_pdf(x=data, mu=init.loc[1, 'mu'], sd=init.loc[1, 'sd'])
    alpha2 = np.array(np.log(bx2[0] * init.loc[1, 'pi']))

    # Initial m values (slightly modified from R code)
    m1_alpha = np.array(max(alpha1, alpha2))
    m2_alpha = np.array(max(alpha1, alpha2))

    for t in range(1, n):
        # Alpha for i=1
        m1_alpha_j1 = (alpha1[t - 1]) + np.log(A_new[0, 0])  # m when j=0 and i=0
        m1_alpha_j2 = (alpha2[t - 1]) + np.log(A_new[1, 0])  # m when j=1 and i=0
        m1_alpha = np.append(m1_alpha, max(m1_alpha_j1, m1_alpha_j2))  # max of m1_j1 and m1_j2
        # calculation for alpha when i=1
        alpha1 = np.append(alpha1, np.log(bx1[t]) + m1_alpha[t] + np.log(np.exp(m1_alpha_j1 - m1_alpha[t]) + np.exp(m1_alpha_j2 - m1_alpha[t])))

        # Alpha for i=2
        m2_alpha_j1 = (alpha1[t - 1]) + np.log(A_new[0, 1])  # m when j=1 and i=2
        m2_alpha_j2 = (alpha2[t - 1]) + np.log(A_new[1, 1])  # m when j=2 and i=2
        m2_alpha = np.append(m2_alpha, max(m2_alpha_j1, m2_alpha_j2))  # max of m2_j1 and m2_j2
        # calculation of alpha when i=2
        alpha2 = np.append(alpha2, np.log(bx2[t]) + m2_alpha[t] + np.log(np.exp(m2_alpha_j1 - m2_alpha[t]) + np.exp(m2_alpha_j2 - m2_alpha[t])))

    # m value for log-likelihood, forward algorithm
    m_alpha_ll = max(alpha1[n-1], alpha2[n-1])
    # Forward algorithm log-likelihood
    fwd_ll = m_alpha_ll + np.log(np.exp(alpha1[n-1] - m_alpha_ll) + np.exp(alpha2[n-1] - m_alpha_ll))
    # package the alpha vectors into a list
    alpha = [alpha1, alpha2]

    return fwd_ll, alpha

def hmm_backward(init, data, A_trans):
    # some initializations and settings
    n = len(data)
    A_new = A_trans

    # Initial values for beta1 and beta2 at t=n

    # Probability density values for the data using distribution 1
    bx1 = hmm_norm_pdf(x=data, mu=init.loc[0, 'mu'], sd=init.loc[0, 'sd'])
    # Probability density values for the data using distribution 2
    bx2 = hmm_norm_pdf(x=data, mu=init.loc[1, 'mu'], sd=init.loc[1, 'sd'])

    beta1 = np.zeros(n)
    beta1[n-1] = (np.log(1))
    beta2 = np.zeros(n)
    beta2[n-1] = (np.log(1))

    for t in reversed(range(0, (n-1))):  # recall that n-2 is actually the second to last position and n-1 is the last position
      # beta for i=1
      m1_beta_j1 = (beta1[t+1] + np.log(A_new[0,0]) + np.log(bx1[t+1]))[0]  # m when j=0 and i=0
      m1_beta_j2 = (beta2[t+1] + np.log(A_new[0,1]) + np.log(bx2[t+1]))[0]  # m when j=1 and i=0
      m1_beta = max(m1_beta_j1, m1_beta_j2)
      beta1[t] = m1_beta + np.log(np.exp(m1_beta_j1 - m1_beta) + np.exp(m1_beta_j2 - m1_beta))

      # beta for i=2
      m2_beta_j1 = (beta1[t+1] + np.log(A_new[1, 0]) + np.log(bx1[t+1]))[0]  # m when j=0 and i=1
      m2_beta_j2 = (beta2[t+1] + np.log(A_new[1, 1]) + np.log(bx2[t+1]))[0] # m when j=1 and i=1
      m2_beta = max(m2_beta_j1, m2_beta_j2)
      beta2[t] = m2_beta + np.log(np.exp(m2_beta_j1 - m2_beta) + np.exp(m2_beta_j2 - m2_beta))

    # first and second parts of m value for log-likelihood backward algorithm
    m_beta_ll1 = (beta1[0] + np.log(init.loc[0, 'pi']) + np.log(bx1[0]))[0]
    m_beta_ll2 = (beta2[0] + np.log(init.loc[0, 'pi']) + np.log(bx2[0]))[0]
    # m value for log likelihood, backward algorithm
    m_beta_ll = max(m_beta_ll1, m_beta_ll2)
    # Backward algorithm log likelihood
    bwd_ll = m_beta_ll + np.log(np.exp(m_beta_ll1 - m_beta_ll) + np.exp(m_beta_ll2 - m_beta_ll))
    # package the beta vectors into a list
    beta = [beta1, beta2]

    return bwd_ll, beta

def hmm_gamma(alpha, beta, n):
    # log gamma z's
    m_gamma = np.maximum(alpha[0] + beta[0], alpha[1] + beta[1])
    log_gamma1 = alpha[0] + beta[0] - m_gamma - np.log(np.exp(alpha[0] + beta[0] - m_gamma) + np.exp(alpha[0] + beta[0] - m_gamma))
    gamma1 = np.exp(log_gamma1)

    z = np.zeros(n) # n is the length of the data
    z_draw = np.random.uniform(low=0, high=1, size=n)
    z[z_draw <= gamma1] = 1
    return z

def hmm_update():
    pass
    return

path = 'output/HMM_data_final.txt'
data = hmm_get_data(path)
init = hmm_intialize(mu_list=[8, 14], sd_list=[5**0.5, 5**0.5], pi_list=[0.5, 0.5])
A_trans = hmm_transition(a_list=[0.8, 0.2, 0.2, 0.8])
fwd, alpha = hmm_forward(init=init, data=data, A_trans=A_trans)
bwd, beta = hmm_backward(init=init, data=data, A_trans=A_trans)
z = hmm_gamma(alpha=alpha, beta=beta, n=len(data))
print(bwd)
print(fwd)
print(fwd-bwd)


"""
-----------------------------------------------------------------------------------------------------------------------
R Code
-----------------------------------------------------------------------------------------------------------------------

hmm_2component = function(hmm_data, num_iter, lag=1, burnin=0){

  # initializations
  n = length(hmm_data)
  mu1_new = 8
  mu2_new = 14
  s1_new = 5
  s2_new = 5
  pi1_new = 0.5
  pi2_new = 1 - pi1_new
  keep = c() # initialize vector to determine retained samples after
  # adjust number of iterations to account for lag and burn-in
  num_iter_tot = num_iter * lag + burnin
  # empty vectors to hold parameter updates
  mu1_vec = c()
  mu2_vec = c()
  s1_vec = c()
  s2_vec = c()
  pi1_vec = c()
  a11_vec = c()
  a22_vec = c()
  # Initial elements of transition matrix
  a11_new = 0.8
  a12_new = 0.2
  a21_new = 0.2
  a22_new = 0.8
  A_new = matrix(c(a11_new, a12_new, a21_new, a22_new), nrow=2,ncol=2, byrow=TRUE)

  for (i in 1:num_iter){
    # flag rows to retain based on lag and burnin
    if ((i%%lag==0)&(i>burnin)){
        keep = c(keep, 1) # if value is 1 then retain the sample
    } else keep = c(keep, 0)
    # -----------------------------------------------------------------------------
    # --------------------FORWARD ALGORITHM ---------------------------------------
    # -----------------------------------------------------------------------------
    # Probability density values for the data using distribution 1
    bx1 =dnorm(hmm_data, mean=mu1_new, sd=sqrt(s1_new))
    alpha1 = c(log(bx1[1] * pi1_init)) # the first entry for alpha1 is bx1
    # Probability density values for the data using distribution 2
    bx2 = dnorm(hmm_data, mean=mu2_new, sd=sqrt(s2_new))
    alpha2 = c(log(bx2[1] * pi2_new)) # the first entry for alpha2 is bx2
    # Initial m values (not exactly correct, but never used)
    m1_alpha = c(max(log(bx1[1]), log(bx2[1])))
    m2_alpha = c(max(log(bx1[1]), log(bx2[1])))

    for (t in 2:n){
        # Alpha for i=1
        m1_alpha_j1 = (alpha1[t-1]) + log(A_new[1,1]) # m when j=1 and i=1
        m1_alpha_j2 = (alpha2[t-1]) + log(A_new[2,1]) # m when j=2 and i=1
        m1_alpha = c(m1_alpha, max(m1_alpha_j1, m1_alpha_j2)) # max of m1_j1 and m1_j2
        # calculation for alpha when i=1
        alpha1 = c(alpha1, log(bx1[t]) + m1_alpha[t] + log(exp(m1_alpha_j1 -m1_alpha[t]) + exp(m1_alpha_j2 - m1_alpha[t])))

        # Alpha for i=2
        m2_alpha_j1 = (alpha1[t-1]) + log(A_new[1,2]) # m when j=1 and i=2
        m2_alpha_j2 = (alpha2[t-1]) + log(A_new[2,2]) # m when j=2 and i=2
        m2_alpha = c(m2_alpha, max(m2_alpha_j1, m2_alpha_j2)) # max of m2_j1 and m2_j2
        # calculation of alpha when i=2
        alpha2 = c(alpha2, log(bx2[t]) + m2_alpha[t] + log(exp(m2_alpha_j1 -m2_alpha[t]) + exp(m2_alpha_j2 - m2_alpha[t])))

      }
    # m value for log-likelihood, forward algorithm
    m_alpha_ll = max(alpha1[n], alpha2[n])
    # Forward algorithm log-likelihood
    fwd_ll = m_alpha_ll + log( exp(alpha1[n] - m_alpha_ll) + exp( alpha2[n] - m_alpha_ll ) )

    # -----------------------------------------------------------------------------
    # --------------------BACKWARD ALGORITHM --------------------------------------
    # -----------------------------------------------------------------------------

    # Initial values for beta1 and beta2 at t=n
    beta1 = numeric(length=n)
    beta1[n] = (log(1))
    beta2 = numeric(length=n)
    beta2[n] = (log(1))

    for (t in (n-1):1){
      # beta for i=1
      m1_beta_j1 = beta1[t+1] + log(A_new[1,1]) + log(bx1[t+1]) # m when j=1 and i=1
      m1_beta_j2 = beta2[t+1] + log(A_new[1,2]) + log(bx2[t+1]) # m when j=2 and i=1
      m1_beta = max(m1_beta_j1, m1_beta_j2)
      beta1[t] = m1_beta + log( exp(m1_beta_j1 - m1_beta) + exp(m1_beta_j2 - m1_beta))

      # beta for i=2
      m2_beta_j1 = beta1[t+1] + log(A_new[2,1]) + log(bx1[t+1]) # m when j=1 and i=2
      m2_beta_j2 = beta2[t+1] + log(A_new[2,2]) + log(bx2[t+1]) # m when j=2 and i=2
      m2_beta = max(m2_beta_j1, m2_beta_j2)
      beta2[t] = m2_beta + log( exp(m2_beta_j1 - m2_beta) + exp(m2_beta_j2 - m2_beta))

    }
    # first and second parts of m value for log-likelihood backward algorithm
    m_beta_ll1 = beta1[1] + log(pi1_new) + log(bx1[1])
    m_beta_ll2 = beta2[1] + log(pi2_new) + log(bx2[1])
    # m value for log likelihood, backward algorithm
    m_beta_ll = max(m_beta_ll1, m_beta_ll2)
    # Backward algorithm log likelihood
    bwd_ll = m_beta_ll + log(exp( m_beta_ll1 - m_beta_ll ) + exp(m_beta_ll2 - m_beta_ll))

    # -----------------------------------------------------------------------------
    # -------------------- GAMMA AND Z---------------------------------------------
    # -----------------------------------------------------------------------------

    # log gamma z's
    m_gamma = max(alpha1 + beta1, alpha2 + beta2)
    log_gamma1 = alpha1 + beta1 - m_gamma - log( exp( alpha1 + beta1 - m_gamma ) + exp( alpha2 + beta2 - m_gamma ))
    gamma1 = exp(log_gamma1)

    z = numeric(length=length(hmm_data))
    z_draw = runif(n=length(hmm_data))
    z[z_draw <= gamma1 ] = 1

    # -----------------------------------------------------------------------------
    # -------------------- UPDATE PARAMETERS --------------------------------------
    # -----------------------------------------------------------------------------

    # Update mu1, mu2, sigma1^2, sigma2^2, pi1, pi2, and transition matrix
    mu1_new = sum(hmm_data[z==1]) / sum(z)
    mu2_new = sum(hmm_data[z==0]) / (length(z) - sum(z))
    sigma1_new = sum( (hmm_data[z==1] - mu1_new)^2 ) / sum(z)
    sigma2_new = sum( (hmm_data[z==0] - mu2_new)^2 ) / (length(z) - sum(z))
    # pi1_new = gamma1[1]
    pi1_new = length(z[z==1]) / length(z)
    pi2_new = 1 - gamma1[1]

    # indicator function for transition matrix
    z1_stay=0
    z1_arrive=0
    for (v in 1:(length(z)-1)){
      if ((z[v]==1) & (z[v+1]==1)){
        z1_stay=z1_stay+1
      }
      if ((z[v]==0) & (z[v+1]==1)){
        z1_arrive=z1_arrive+1
      }
    }
    # updated elements for transition matrix
    a11_new = z1_stay / sum(z)
    a12_new = 1 - a11_new
    a21_new = z1_arrive / length(z[z==0])
    a22_new = 1 - a21_new
    # final updated transition matrix
    A_new = matrix(c(a11_new, a12_new, a21_new, a22_new), nrow=2,ncol=2, byrow=TRUE)
    # save the parameters to their vectors
    mu1_vec = c(mu1_vec, mu1_new)
    mu2_vec = c(mu2_vec, mu2_new)
    s1_vec = c(s1_vec, sigma1_new)
    s2_vec = c(s2_vec, sigma2_new)
    pi1_vec = c(pi1_vec, pi1_new)
    a11_vec = c(a11_vec, a11_new)
    a22_vec = c(a22_vec, a22_new)
  }

  df = data.frame(
    mu1 = mu1_vec,
    mu2 = mu2_vec,
    s1 = s1_vec,
    s2 = s2_vec,
    pi1 = pi1_vec,
    a11 = a11_vec,
    a22 = a22_vec,
    keep = keep
  )
  # final output dataframe removing the lag and burn-in samples
  out = df[df$keep==1,]
  return(df)
}

"""