#Environment generator for Social Generalization 
#Charley Wu (Oct 2025)

rm(list=ls())
packages<-c("plyr", 'cowplot', 'rdist', 'ggplot2' , 'viridis', 'jsonlite')
invisible(lapply(packages, require, character.only = TRUE))

#Globally fixed prameters
gridSize <- 11
xstar <- expand.grid(x=1:gridSize, y = 1:gridSize) #input space
lambda <- 2 #length scale

#########################################################################################################################
# Gaussian Process functions
#########################################################################################################################
# kernel function
#l =  length scale (aka lambda)

rbf_D <- function(X1, X2=NULL,l=1){
  if (is.null(X2)){
    D <- pdist(X1)^2
  }else{
    D <- cdist(X1, X2)^2
  }
  Sigma <- exp(-D/(2*l^2))
}

#Gaussian Process function
#lambda is length scale
#eps is error variance
#k allows for selecting other kernels
#full_cov is whether to return a dataframe of the mean and variance of each option, or to return the full covariance matrix (for sampling)
gpr <- function(Xstar,X,Y, lambda, eps = sqrt(.Machine$double.eps), k = rbf_D, full_cov = F ){
  #Compute the covariance between observed inputs
  K <- k(X,X,lambda) #K(x,x') for each pair of observed inputs in X
  KK <- K + diag(eps, nrow(K)) #(K + noise * I)
  KK.inv <- chol2inv(chol(KK)) #Invert matrix using Cholesky decomposition
  Ky <- KK.inv %*% Y # #times y
  if(!full_cov){ #return only the mean and variance vectors
    result <- sapply(Xstar, function(x_i){ 
      #Compute covariance of observed inputs with target space (Xstar)
      Kstar <- k(X, x_i, lambda)
      Kstarstar <- k(x_i,x_i,lambda)  #Covariance of Xstar with itself
      #Compute posterior as a mean vector and a variance vector
      mu <-t(Kstar)  %*% Ky #get mean vector
      var <- Kstarstar - (t(Kstar) %*% KK.inv %*% Kstar) #get covariance
      cbind(mu,var)
    })
    prediction <- as.data.frame(t(result))
    colnames(prediction) <- c('mu', 'var')
    return(prediction) #return it as a data farm
  }else{#return the full covariance matrix
    #Compute covariance of observed inputs with target space (Xstar)
    Kstar <- k(X, Xstar, lambda)
    Kstarstar <- k(Xstar,Xstar,lambda)  #Covariance of Xstar with itself
    #Compute posterior as a mean vector and a variance vector
    mu <-t(Kstar)  %*% Ky #get mean vector
    cov <- Kstarstar - (t(Kstar) %*% KK.inv %*% Kstar) 
    return(list(mu = mu, cov = cov))
  }
}

#Minmax scaling to 0-1
normalize <- function(x){(x-min(x))/(max(x)-min(x))}
#########################################################################################################################
# Step 0: Parent environments
# Sample spatially correlated environments from the GP prior
#########################################################################################################################
#Parameters
n_envs <- 40
# compute kernel on pairwise values
Sigma <- rbf_D(xstar,l=lambda)
# sample from multivariate normal with mean zero, sigma = sigma
Z <- MASS::mvrnorm(n_envs,rep(0,dim(Sigma)[1]), Sigma) #Sample a single canonical function 

environmentList <- list()
plot_list = list()
for (i in 1:n_envs){
  z <- normalize(Z[i,]) #scale to 0 and 1
  M <- data.frame(x1 = xstar$x, x2 = xstar$y, payoff = z)
  environmentList[[i]] <- M #add to list
  #plot each env
  plot_list[[i]] <- ggplot(M, aes(x = x1, y = x2, fill = payoff ))+
    geom_tile()+ theme_void() + scale_fill_viridis(name='Payoff') + theme(legend.position='none') + 
    ggtitle(bquote(M[.(i)]^0))
}
#Save plots
payoffplots <- cowplot::plot_grid(plotlist = plot_list, ncol = 8)
ggsave('plots/M0exp_c01_demo.pdf', payoffplots, width = 12, height = 8)
#Save environments
write_json(environmentList, 'environments/M0exp_c01_demo_parent.json')


#########################################################################################################################
# Step 1) Correlated environment generation (as in the original paper)
# Sample a canonical function from the GP prior, and then use that function as the mean function
# Then sample individual payoff functions from a GP where the mean is defined but the variance is still the prior variance
#########################################################################################################################
#Simulation parameters
genNum <- 10000
n_players <- 4
n_envs <- 40
correlationThreshold <- .6 #At least .6
tolerance <- .05
childNames = c('A_exp_demo', 'B_exp_demo', 'C_exp_demo', 'D_exp_demo')

# compute kernel on pairwise values
Sigma_social <- rbf_D(xstar,l=lambda)

#M0 generated above are the canonical environments
M <-fromJSON("environments/M0exp_c01_demo_parent.json", flatten=TRUE) #load from above

childEnvList = list(A=list(), B=list(), C=list(), D=list())
plot_list = list(list(), list(), list(), list())
#Sample functions from the new prior mean is defined by the canonical environment
for (i in 1:n_envs){
  Z_n <- MASS::mvrnorm(genNum,M[[i]][,'payoff'], Sigma_social, ) #generate many candidates
  cors<- sapply(1:genNum, FUN = function(k) cor(M[[i]][,'payoff'], Z_n[k,])) #compute correlations with canonical environment
  #remove environments with correlations lower than threshold with canonical
  Z_n <- Z_n[cors>correlationThreshold,] #
  #Try to find a set of 4 environments, where all envs have above threshold correlations amongst each other
  found <- FALSE 
  while (found==FALSE){
    candidates <- sample(1:nrow(Z_n), size = n_players)
    cors <- c()
    for (k in 2:n_players){
      cors <- c(cors,cor(Z_n[candidates[1],],Z_n[candidates[k],]))
    }
    #Check that all correlations fall within the correlation threshold with A? tolerance
    checked <- sum((cors>=(correlationThreshold-tolerance)) & (cors<=(correlationThreshold+tolerance)))
    if (checked==3){ #now repeat for environment B (with C&D, no need to check AB again --> 2)
      for (k in 3:n_players){
        cors <- c(cors,cor(Z_n[candidates[2],],Z_n[candidates[k],]))
      }
      checked <- sum((cors>=(correlationThreshold-tolerance)) & (cors<=(correlationThreshold+tolerance)))
      # if (checked==5){
      #   #found <- T
      #   #print("found one")
      # }
      if (checked==5){#finally, check correlation of C&D
        cors <- c(cors,cor(Z_n[candidates[3],],Z_n[candidates[4],]))
      }
      checked <- sum((cors>=(correlationThreshold-tolerance)) & (cors<=(correlationThreshold+tolerance)))
      if (checked==6){ #all 6 combinations checked --> conditions fulfilled, add to list
        found<-TRUE
        print("found one")
      }
    }
  }
  Z_n <- Z_n[candidates,]
  for (j in 1:n_players){
    Z_j <- normalize(Z_n[j,])
    entry <- data.frame(x1=xstar$x, x2 = xstar$y, payoff=Z_j)
    childEnvList[[childNames[j]]][[i]] <- entry #add to list
    #plot each env
    plot_list[[j]][[i]] <- ggplot(entry, aes(x = x1, y = x2, fill = payoff ))+
      geom_tile()+ theme_void() + scale_fill_viridis(name='Payoff') + theme(legend.position='none') + 
      ggtitle(bquote(.(childNames[[j]])[.(i)]^1))
  }
}

 #Save plots
for (child in childNames){
  i <- match(child, childNames)
  payoffplots <- cowplot::plot_grid(plotlist = plot_list[[i]], ncol = 8)
  ggsave(paste0('plots/', child,'_c01_demo.pdf'), payoffplots, width = 12, height = 8)  
}

#Save environments
for (child in childNames){
  write_json(childEnvList[[child]], paste0('environments/', child,'_c01_demo.json'))
}
 


#########################################################################################################################
# Step 2) Correlated environment generation with unequal similarity to target
# For each canonical environment M[[i]] (the target, A), generate B, C, D such that
# cor(A,B) ~= target_r_with_A[1], cor(A,C) ~= target_r_with_A[2], cor(A,D) ~= target_r_with_A[3].
# Outputs go to dedicated directories to avoid overlapping your original results.
#########################################################################################################################

plots_dir <- 'plots_unequal'
env_dir   <- 'environments_unequal'

genNum <- 10000
n_players <- 4                # A (target), B, C, D
n_envs <- 40                  # number of canonical environments to process
childNames <- c('A_exp_demo', 'B_exp_demo', 'C_exp_demo', 'D_exp_demo')

# Desired correlations with the target (canonical) environment A = M[[i]].
# Order corresponds to B, C, D respectively.
target_r_with_A <- c(0.6, 0.2, 0.0)   # pairwise target correlations
tol_A           <- c(0.05, 0.05, 0.05)   # per-target tolerances (same order as above)

# Kernel already defined above; reuse the same social kernel
Sigma_social <- rbf_D(xstar, l = lambda)

# Load canonical environments (from your Step 0)
M <- fromJSON("environments/M0exp_c01_demo_parent.json", flatten = TRUE)

# Containers
childEnvList_unequal <- list(A = list(), B = list(), C = list(), D = list())
plot_list_unequal    <- list(list(), list(), list(), list())

# Helper: shortlist indices near a target correlation; fallback to nearest topK if empty
in_band <- function(cvec, center, halfw) which((cvec >= (center - halfw)) & (cvec <= (center + halfw)))
fallback_if_empty <- function(idx, cvec, center, K = 200){
  if (length(idx) == 0){
    ord <- order(abs(cvec - center))
    return(ord[seq_len(min(K, length(ord)))])
  } else {
    idx_sub <- idx[order(abs(cvec[idx] - center))]
    return(idx_sub[seq_len(min(K, length(idx_sub)))])
  }
}

# Main loop: for each canonical environment (target)
for (i in 1:n_envs){
  A_vec <- M[[i]][,'payoff']  # target vector (already 0-1 scaled in your Step 0)
  
  # Sample candidate functions with GP mean = target (same as your original)
  Z_n <- MASS::mvrnorm(genNum, A_vec, Sigma_social)
  
  # Correlations with A for all candidates
  cors_with_A <- as.numeric(cor(t(Z_n), A_vec))
  
  # Build shortlists for B, C, D based on A–X target bands
  idx_B <- in_band(cors_with_A, target_r_with_A[1], tol_A[1])
  idx_C <- in_band(cors_with_A, target_r_with_A[2], tol_A[2])
  idx_D <- in_band(cors_with_A, target_r_with_A[3], tol_A[3])
  
  # Relax if needed + limit shortlist size for speed
  topK <- 200
  idx_B <- fallback_if_empty(idx_B, cors_with_A, target_r_with_A[1], topK)
  idx_C <- fallback_if_empty(idx_C, cors_with_A, target_r_with_A[2], topK)
  idx_D <- fallback_if_empty(idx_D, cors_with_A, target_r_with_A[3], topK)
  
  # Choose distinct indices (greedy by closeness to targets)
  # You can replace this with a tiny grid search if you want to optimize B–C–D mutual correlations.
  pick_one <- function(idx, used) setdiff(idx, used)[1]
  used <- integer(0)
  b <- pick_one(idx_B, used); used <- c(used, b)
  c <- pick_one(idx_C, used); used <- c(used, c)
  d <- pick_one(idx_D, used); used <- c(used, d)
  
  chosen_idx <- c(b, c, d)
  
  # Normalize and save A (the target) + B/C/D (the chosen)
  # A is just the canonical environment itself
  Z_A <- normalize(A_vec)
  entry_A <- data.frame(x1 = xstar$x, x2 = xstar$y, payoff = Z_A)
  childEnvList_unequal[[childNames[1]]][[i]] <- entry_A
  
  # Plot A
  plot_list_unequal[[1]][[i]] <-
    ggplot(entry_A, aes(x = x1, y = x2, fill = payoff )) +
    geom_tile() + theme_void() + scale_fill_viridis(name='Payoff') + theme(legend.position='none') +
    ggtitle(bquote(.(childNames[[1]])[.(i)]^"unequal"))
  
  # Now B, C, D from chosen candidates
  for (j in 2:n_players){
    Z_j <- normalize(Z_n[chosen_idx[j-1 - 1 + 1], ])  # map j=2->b, j=3->c, j=4->d
    entry <- data.frame(x1 = xstar$x, x2 = xstar$y, payoff = Z_j)
    childEnvList_unequal[[childNames[j]]][[i]] <- entry
    
    # Plot each env
    plot_list_unequal[[j]][[i]] <-
      ggplot(entry, aes(x = x1, y = x2, fill = payoff )) +
      geom_tile() + theme_void() + scale_fill_viridis(name='Payoff') + theme(legend.position='none') +
      ggtitle(bquote(.(childNames[[j]])[.(i)]^"unequal"))
  }
}

# Save plots (to non-overlapping directory)
for (child in childNames){
  idx <- match(child, childNames)
  payoffplots <- cowplot::plot_grid(plotlist = plot_list_unequal[[idx]], ncol = 8)
  ggsave(file.path(plots_dir, paste0(child, '_c01_unequal_demo.pdf')), payoffplots, width = 12, height = 8)
}

# Save environments (to non-overlapping directory)
for (child in childNames){
  jsonlite::write_json(childEnvList_unequal[[child]], file.path(env_dir, paste0(child, '_c01_unequal_demo.json')))
}
