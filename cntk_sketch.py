import numpy as np
import torch
import torch.fft

from ntk_sketch import quadprog_solve_qp


def krelu(q, h):
    alpha_ = -1.0
    for i in range(h):
        alpha_ = (2.0*alpha_ + (np.sqrt(1-alpha_**2) + alpha_*(np.pi - np.arccos(alpha_)))/np.pi)/3.0
    y = (np.sqrt(1-np.linspace(alpha_, 1.0, num=201)**2) + np.linspace(alpha_, 1.0, num=201)*(np.pi - np.arccos(np.linspace(alpha_, 1.0, num=201))))/np.pi
        
    Z = np.zeros((201,q+1))
    Z[:,0] = np.ones(201)
    for i in range(q):
        Z[:,i+1] = Z[:,i] * np.linspace(alpha_, 1.0, num=201)
        
    w = y
    U = Z.T

    coeff = quadprog_solve_qp(np.dot(U, U.T), -np.dot(U,w) , np.concatenate((Z[0:200,:]-Z[1:201,:], -np.eye(q+1)),axis=0), np.zeros(q+201), Z[200,:][np.newaxis,:],y[200])
    coeff[coeff < 0.00001] = 0
    
    return coeff


def kdotrelu(q,h):
    alpha_ = -1.0
    for i in range(h):
        alpha_ = (1.0*alpha_ + (np.sqrt(1-alpha_**2) + alpha_*(np.pi - np.arccos(alpha_)))/np.pi)/2.0
    y = (np.pi - np.arccos(np.linspace(alpha_, 1.0, num=201)))/np.pi
    
    Z = np.zeros((201,q+1))
    Z[:,0] = np.ones(201)
    for i in range(q):
        Z[:,i+1] = Z[:,i] * np.linspace(alpha_, 1.0, num=201)
    
    
    weight_ = np.linspace(0.0, 1.0, num=201)**2 + 1/2
    w = y * weight_
    U = Z.T * weight_

    coeff = quadprog_solve_qp(np.dot(U, U.T), -np.dot(U,w) , np.concatenate((Z[0:200,:]-Z[1:201,:], -np.eye(q+1)),axis=0), np.zeros(q+201))
    coeff[coeff < 0.00001] = 0
    
    return coeff


def TSRHTCmplx(X1, X2, P, D):
    D1 = D[0,:].unsqueeze(0).unsqueeze(2).unsqueeze(3)
    D2 = D[1,:].unsqueeze(0).unsqueeze(2).unsqueeze(3)
    
    Xhat1 = torch.fft.fftn(X1 * D1, dim=1)[:,P[0,:],:,:]
    Xhat2 = torch.fft.fftn(X2 * D2, dim=1)[:,P[1,:],:,:]
    
    Y = np.sqrt(1/P.shape[1])*(Xhat1 * Xhat2)
    
    return Y


def SRHTCmplx_Stndrd(X, P, D):    
    return np.sqrt(1/len(P))* torch.fft.fftn(X * D.unsqueeze(0).unsqueeze(2).unsqueeze(3), dim=1)[:,P,:,:]


def Conv2DTensor(filt_size, X):
    
    Sigma_b = X[:, :, 0:X.shape[2]-filt_size+1]
    
    for b in range(1,filt_size):
        Sigma_b = Sigma_b + X[:, :, b:X.shape[2]-filt_size+1+b]
        
    Sigma = Sigma_b[:, 0:Sigma_b.shape[1]-filt_size+1, :]
    
    for a in range(1,filt_size):
        Sigma = Sigma + Sigma_b[:, a:Sigma_b.shape[1]-filt_size+1+a, :]
        
    return Sigma


def DirectSumTensor(filt_size, X, dev):
    
    Sigma_b = torch.zeros((X.shape[0], filt_size*X.shape[1], X.shape[2], X.shape[3]-filt_size+1), dtype=X.dtype, device=dev)
    
    for b in range(filt_size):
        Sigma_b[:, b*X.shape[1]:(b+1)*X.shape[1], :, :] = X[:, :, :, b:X.shape[3]-filt_size+1+b]
        
    Sigma = torch.zeros((Sigma_b.shape[0], filt_size*Sigma_b.shape[1], Sigma_b.shape[2]-filt_size+1, Sigma_b.shape[3]), dtype=X.dtype, device=dev)
    
    for a in range(filt_size):
        Sigma[:, a*Sigma_b.shape[1]:(a+1)*Sigma_b.shape[1], :, :] = Sigma_b[:, :, a:Sigma_b.shape[2]-filt_size+1+a, :]
        
    return Sigma


class CNTKSketch:
    def __init__(self, filt_size, Channs, m, q, L, dev):
        self.filt_size = filt_size
        self.Channs = Channs
        self.m = m
        self.q = q
        self.L = L
        self.device_ = dev

        self.Tree_D = [0 for i in range(self.L)]
        self.Tree_P = [0 for i in range(self.L)]
        
        self.ConvD = [0 for i in range(self.L)]
        self.ConvP = [0 for i in range(self.L)]
        
        self.poly_D = [0 for i in range(self.L)]
        self.poly_P = [0 for i in range(self.L)]
        
        self.phidot_D = [0 for i in range(self.L)]
        self.phidot_P = [0 for i in range(self.L)]
        
        self.psi_T_D = [0 for i in range(self.L)]
        self.psi_T_P = [0 for i in range(self.L)]
        
        self.psi_D = [0 for i in range(self.L)]
        self.psi_P = [0 for i in range(self.L)]
        
        self.psiConv_D = [0 for i in range(self.L)]
        self.psiConv_P = [0 for i in range(self.L)]

        for i in range(self.L):
            if i == 0:
                self.Tree_D[i], self.Tree_P[i] = self.TensorInit(self.Channs*self.filt_size**2 , int(self.m/4))
            else:
                self.Tree_D[i], self.Tree_P[i] = self.TensorInit(int(self.m/2), int(self.m/4))
            
                self.ConvD[i] = torch.from_numpy(np.random.choice((-1,1), int(self.m/8)*self.filt_size**2)).to(self.device_)
                self.ConvP[i] = torch.from_numpy(np.random.choice(int(self.m/8)*self.filt_size**2, int(self.m/2 - np.ceil(self.filt_size**2 / self.q)))).to(self.device_)
            
                self.poly_D[i] = torch.from_numpy(np.random.choice((-1,1), self.q * len(self.ConvP[i]) + self.filt_size**2)).to(self.device_)
                self.poly_P[i] = torch.from_numpy(np.random.choice(len(self.poly_D[i]), int(self.m/2))).to(self.device_)
            
                self.phidot_D[i] = torch.from_numpy(np.random.choice((-1,1), self.q*int(self.m/8))).to(self.device_)
                self.phidot_P[i] = torch.from_numpy(np.random.choice(self.q*int(self.m/8), int(self.m/2)-1)).to(self.device_)
            
                self.psi_T_D[i] = torch.from_numpy(np.random.choice((-1,1), (2,int(self.m/2)))).to(self.device_)
                self.psi_T_P[i] = torch.from_numpy(np.random.choice(int(self.m/2), (2,int(self.m/2)))).to(self.device_)
            
                if i < self.L-1:
                    self.psi_D[i] = torch.from_numpy(np.random.choice((-1,1), int(self.m/2 + self.q*self.m/8))).to(self.device_)
                    self.psi_P[i] = torch.from_numpy(np.random.choice(int(self.m/2 + self.q*self.m/8), int(self.m/4-1))).to(self.device_)
                
                    self.psiConv_D[i] = torch.from_numpy(np.random.choice((-1,1), int(self.m/4)*self.filt_size**2)).to(self.device_)
                    self.psiConv_P[i] = torch.from_numpy(np.random.choice(int(self.m/4)*self.filt_size**2, int(self.m/2))).to(self.device_)

    def TensorInit(self, d, m):
    
        Tree_D = [0 for i in range((self.q-1).bit_length())]
        Tree_P = [0 for i in range((self.q-1).bit_length())]
    
        m_=int(m/2)
        q_ = int(self.q/2)
        for i in range((self.q-1).bit_length()):
            if i == 0:
                Tree_P[i] = torch.from_numpy(np.random.choice(d, (q_,2,m_))).to(self.device_)
                Tree_D[i] = torch.from_numpy(np.random.choice((-1,1), (q_,2,d))).to(self.device_)
            else:
                Tree_P[i] = torch.from_numpy(np.random.choice(m_, (q_,2,m_))).to(self.device_)
                Tree_D[i] = torch.from_numpy(np.random.choice((-1,1), (q_,2,m_))).to(self.device_)
            q_ = int(q_/2)
        
        return Tree_D, Tree_P

    
    def TensorSketchT(self, X, h):
        n=X.shape[0]
        lgq = len(self.Tree_D[h])
        V = [0 for i in range(lgq)]
        E1 = torch.cat((torch.ones((n, 1, X.shape[2], X.shape[3]), device=self.device_), torch.zeros((n, X.shape[1]-1, X.shape[2], X.shape[3]), device=self.device_)), 1)
    
        for i in range(lgq):
            q = self.Tree_D[h][i].shape[0]
            V[i] = torch.zeros((q, n, self.Tree_P[h][i].shape[2], X.shape[2], X.shape[3]), dtype=torch.cfloat, device=self.device_)
            for j in range(q):
                if i == 0:
                    V[i][j,:,:,:,:] = TSRHTCmplx(X, X, self.Tree_P[h][i][j,:,:], self.Tree_D[h][i][j,:,:])
                else:
                    V[i][j,:,:,:,:] = TSRHTCmplx(V[i-1][2*j,:,:,:,:], V[i-1][2*j+1,:,:,:,:], self.Tree_P[h][i][j,:,:], self.Tree_D[h][i][j,:,:])
    
        U = [0 for i in range(2**lgq)]
        U[0] = V[lgq-1][0,:,:,:,:].detach().clone()
    
        for j in range(1,len(U)):
            p = int((j-1)/2)
            for i in range(lgq):
                if j%(2**(i+1)) == 0 :
                    V[i][p,:,:,:,:] = torch.cat((torch.ones((n, 1, X.shape[2], X.shape[3]), device=self.device_), torch.zeros((n, V[i].shape[2]-1, X.shape[2], X.shape[3]), device=self.device_)), 1)
                else:
                    if i == 0:
                        V[i][p,:,:,:,:] = TSRHTCmplx(X, E1, self.Tree_P[h][i][p,:,:], self.Tree_D[h][i][p,:,:])
                    else:
                        V[i][p,:,:,:,:] = TSRHTCmplx(V[i-1][2*p,:,:,:,:], V[i-1][2*p+1,:,:,:,:], self.Tree_P[h][i][p,:,:], self.Tree_D[h][i][p,:,:])
                p = int(p/2)
            U[j] = V[lgq-1][0,:,:,:,:].detach().clone()
        
        return U


def OblvFeatCNTK(cntk_sketch, X):
    n = X.shape[0]
    q = cntk_sketch.q
    filt_size = cntk_sketch.filt_size
    
    L = cntk_sketch.L
    Normalizer = [0 for i in range(L)]
    for h in range(L):
        if h==0:
            Normalizer[h] = torch.clamp(Conv2DTensor(filt_size, torch.sum(X**2 , dim=1)), min=0.00000001)
        else:
            Normalizer[h] = (1/filt_size**2) * Conv2DTensor(filt_size, Normalizer[h-1])

    mu_ = DirectSumTensor(filt_size, X, cntk_sketch.device_) / torch.sqrt(Normalizer[0]).unsqueeze(1)
    
    Z = cntk_sketch.TensorSketchT(mu_, 0)

    coeff_krelu = krelu(q,0)

    Conv_Z = torch.zeros((n, q*len(cntk_sketch.ConvP[1])+filt_size**2, mu_.shape[2]-filt_size+1, mu_.shape[3]-filt_size+1), dtype=torch.cfloat, device = cntk_sketch.device_)
    mu_ = 0
    
    Conv_Z[:, 0:filt_size**2, :, :] = DirectSumTensor(filt_size, (np.sqrt(coeff_krelu[0])/filt_size) * torch.sqrt(Normalizer[0]).unsqueeze(1), cntk_sketch.device_)
    for i in range(q):
        Conv_Z[:, i*len(cntk_sketch.ConvP[1])+filt_size**2:(i+1)*len(cntk_sketch.ConvP[1])+filt_size**2, :, :] = SRHTCmplx_Stndrd(DirectSumTensor(filt_size, (np.sqrt(coeff_krelu[i+1])/filt_size) * torch.sqrt(Normalizer[0]).unsqueeze(1) * Z[q-i-1], cntk_sketch.device_), cntk_sketch.ConvP[1], cntk_sketch.ConvD[1])
    
    psi_ = SRHTCmplx_Stndrd(Conv_Z, cntk_sketch.poly_P[1], cntk_sketch.poly_D[1])
    
    Conv_Z = 0
    mu_ = psi_ / torch.sqrt(Normalizer[1]).unsqueeze(1)

    for h in range(1,L-1):
        coeff_krelu = krelu(q,h)
        coeff_kreludot = kdotrelu(q,h)
        
        Z = cntk_sketch.TensorSketchT(mu_,h)

        Conv_Z = torch.zeros((n, q*len(cntk_sketch.ConvP[h+1])+filt_size**2, mu_.shape[2]-filt_size+1, mu_.shape[3]-filt_size+1), dtype=torch.cfloat, device=cntk_sketch.device_)
        mu_ = 0
        Conv_Z[:, 0:filt_size**2, :, :] = DirectSumTensor(filt_size, (np.sqrt(coeff_krelu[0])/filt_size) * torch.sqrt(Normalizer[h]).unsqueeze(1), cntk_sketch.device_)
        for i in range(q):
            Conv_Z[:, i*len(cntk_sketch.ConvP[h+1])+filt_size**2:(i+1)*len(cntk_sketch.ConvP[h+1])+filt_size**2, :, :] = SRHTCmplx_Stndrd(DirectSumTensor(filt_size, (np.sqrt(coeff_krelu[i+1])/filt_size) * torch.sqrt(Normalizer[h]).unsqueeze(1) * Z[q-i-1], cntk_sketch.device_), cntk_sketch.ConvP[h+1], cntk_sketch.ConvD[h+1])
            
        mu_ = SRHTCmplx_Stndrd(Conv_Z, cntk_sketch.poly_P[h+1], cntk_sketch.poly_D[h+1]) / torch.sqrt(Normalizer[h+1]).unsqueeze(1)
        Conv_Z = 0

        Conv_Zdot = torch.zeros((n, len(cntk_sketch.phidot_D[h]), psi_.shape[2], psi_.shape[3]), dtype=torch.cfloat, device=cntk_sketch.device_)
        for i in range(q):
            Conv_Zdot[:, i*Z[i].shape[1]:(i+1)*Z[i].shape[1], :, :] = (np.sqrt(coeff_kreludot[i+1])/filt_size) * Z[q-i-1]
        
        Conv_Zdot = SRHTCmplx_Stndrd(Conv_Zdot, cntk_sketch.phidot_P[h], cntk_sketch.phidot_D[h])
        
        psi_ = TSRHTCmplx(psi_, torch.cat(((np.sqrt(coeff_kreludot[0])/filt_size)*torch.ones((n, 1, psi_.shape[2], psi_.shape[3]), device=cntk_sketch.device_), Conv_Zdot), 1), cntk_sketch.psi_T_P[h], cntk_sketch.psi_T_D[h])
        Conv_Zdot = 0
        
        Conv_phi_psi = torch.zeros((n, len(cntk_sketch.psi_D[h]), psi_.shape[2], psi_.shape[3]), dtype=torch.cfloat, device=cntk_sketch.device_)

        for i in range(q):
            Conv_phi_psi[:, i*Z[i].shape[1]:(i+1)*Z[i].shape[1], :, :] = (np.sqrt(coeff_krelu[i+1])/filt_size) * torch.sqrt(Normalizer[h]).unsqueeze(1) * Z[q-i-1]
        
        Conv_phi_psi[:, q*Z[0].shape[1]:Conv_phi_psi.shape[1], :, :] = psi_
        Conv_phi_psi = SRHTCmplx_Stndrd(Conv_phi_psi, cntk_sketch.psi_P[h], cntk_sketch.psi_D[h])
        psi_ = SRHTCmplx_Stndrd(DirectSumTensor(filt_size, torch.cat((Conv_phi_psi, (np.sqrt(coeff_krelu[0])/filt_size) * torch.sqrt(Normalizer[h]).unsqueeze(1)),1), cntk_sketch.device_), cntk_sketch.psiConv_P[h], cntk_sketch.psiConv_D[h])
        Conv_phi_psi = 0
        
    coeff_kreludot = kdotrelu(q,L-1)

    Z = cntk_sketch.TensorSketchT(mu_, L-1)
    Conv_Zdot = torch.zeros((n, len(cntk_sketch.phidot_D[L-1]), psi_.shape[2], psi_.shape[3]), dtype=torch.cfloat, device=cntk_sketch.device_)
    for i in range(q):
        Conv_Zdot[:, i*Z[i].shape[1]:(i+1)*Z[i].shape[1], :, :] = (np.sqrt(coeff_kreludot[i+1])/filt_size) * Z[q-i-1]
    
    Conv_Zdot = SRHTCmplx_Stndrd(Conv_Zdot, cntk_sketch.phidot_P[L-1], cntk_sketch.phidot_D[L-1])
    phi_dot = torch.cat(((np.sqrt(coeff_kreludot[0])/filt_size)*torch.ones((n, 1, psi_.shape[2], psi_.shape[3]), device=cntk_sketch.device_), Conv_Zdot), 1)
    Conv_Zdot = 0
    
    psi_ = TSRHTCmplx(psi_, phi_dot, cntk_sketch.psi_T_P[L-1], cntk_sketch.psi_T_D[L-1])
    
    psi_ = torch.mean(psi_, [2,3])
    return torch.cat((psi_.real, psi_.imag), 1)
