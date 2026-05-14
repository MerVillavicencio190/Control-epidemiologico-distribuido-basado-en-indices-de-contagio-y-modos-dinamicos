#   
#  Creado por: María Fernanda García Villavicencio
#  El modelo fue diseñado en base al funcionamiento de la 
#  librería Epinetwork

class VectorBorne_Endemic(Model):
    def __init__(self,n=1,*,params = np.full((8),None) , network=MobilityNetwork, control = noControl()):
         #Number of patches, parameter of a patch, mobility network, conctrol class
        super().__init__(n=n,params=params,network=network,control=control)
        self.number_of_variables = 5
        self.final_size_presition = 0.0001
        self.final_size_max_iterations = 2000000
        #Dealing with params
        #Asumiremos que los parámetros se dan por columna 

        if params.ndim == 1:
            params = np.tile(params, (n,1))
 

        #Ahora Params puede tener diferentes valores por parche
        self.zeta = params[:, 0]      
        self.mu_h = params[:, 1]
        self.mu_v = params[:, 2]
        self.beta_h = params[:, 3]
        self.beta_v = params[:, 4]
        self.gamma = params[:, 5]        
        self.alpha = params[:, 6]
        self.carry = params[:, 7]

    def system(self, t, yv):
        y = yv.reshape(self.number_of_patches,self.number_of_variables)

        S_h = y[:,0]
        I_h = y[:,1]
        R_h = y[:,2]
        S_v = y[:,3]
        I_v = y[:,4]
        
        Nh = S_h + I_h + R_h
        Nv = S_v + I_v
        

        P = Nh.dot(self.p.matrix) #Esto es Wj
        incidence_h = self.p.matrix.dot(I_v/P)
        incidence_v = self.p.matrix.dot(I_h/P)
             
        dS_h = self.zeta - self.mu_h*S_h - self.beta_h*S_h*incidence_h
        dI_h = self.beta_h*S_h*incidence_h - (self.gamma + self.mu_h)*I_h
        dR_h = self.gamma*I_h - self.mu_h*R_h
        dS_v = self.alpha*Nv*(1-(Nv/self.carry)) - self.mu_v*S_v - self.beta_v*S_v*incidence_v
        dI_v = self.beta_v*S_v*incidence_v - self.mu_v*I_v

        return np.array([dS_h,dI_h,dR_h,dS_v,dI_v]).T.flatten()

    
    def  set_patches_params(self,params,No_patches=None):
        if (No_patches == None) :
            n = self.number_of_patches
        else :
            n = No_patches
            if (self.number_of_patches != n):
                self.number_of_patches = n
                print("Caution : Number of patches has changed")

        self.zeta =  np.full((n),params[0])
        self.mu_h = np.full((n),params[1])
        self.mu_v = np.full((n),params[2])
        self.beta_h = np.full((n),params[3])
        self.beta_v = np.full((n),params[4])
        self.gamma = np.full((n),params[5])        
        self.alpha = np.full((n),params[6])
        self.carry = np.full((n),params[7])

    def get_indices(self, y):

        S_h = y[:,0]
        I_h = y[:,1]
        R_h = y[:,2]
        S_v = y[:,3]
        I_v = y[:,4]

        Nh = S_h + I_h + R_h
        Nv = S_v + I_v

        return self.local_final_size_sup(Nh,Nh,Nv)