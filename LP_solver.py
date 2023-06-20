import numpy as np
from scipy import linalg

threash = 50
epsilon = 0.001


# Ax<=b, obgectice cx
def LP(A, b, c, rule):
    i = 0
    bsize = b.size
    B = np.asmatrix(np.identity(bsize))
    An = A
    Xbs = b.copy()
    Cb = np.matrix(np.zeros(bsize))
    Cn = np.matrix(c)
    asize = A.shape[1]
    Xn = np.matrix(range(0, asize))
    Xb = np.matrix(range(asize, asize + bsize))
    Eta = []
    eta = np.asmatrix(np.identity(bsize))
    p, l, u = linalg.lu(B)
    u = np.asmatrix(linalg.inv(u))
    l = np.asmatrix(linalg.inv(l))
    p = np.asmatrix(linalg.inv(p))
    while True:

        y = Btran(p, l, u, Eta, Cb)
        sev = EnteringVariable(An, Cn, Xn, y, rule)
        if sev == -1:
            Xf = np.matrix(np.zeros(asize + bsize))
            for j in range(0, bsize):
                Xf[0, Xb[0, j]] = Xbs[0, j]
            return Xf[0,:].tolist()[0],np.matmul(Cb, Xbs.T)[0,0],i
        #המקדמים של ה ev
        d = Ftran(p, l, u, Eta, An[:, sev])
        t, slv = LeavingVariable(Xbs, d)
        if t == -1:
            return "unbounded"

        # swap
        eta[:, slv] = d.T
        Eta.append((eta, slv))
        eta = np.asmatrix(np.identity(bsize))

        temp=An[:, sev].copy()
        An[:, sev]=B[:, slv].copy()
        B[:, slv]=temp.copy()

        Xb[0, slv], Xn[0, sev] = Xn[0, sev], Xb[0, slv]

        Cb[0, slv], Cn[0, sev] = Cn[0, sev], Cb[0, slv]
        # swap(B[ev],An[lv])
        Xbs -= t * np.asmatrix(d)
        Xbs[0, slv] = t

        i += 1
        if (i > threash) or not (np.allclose(np.matmul(B,Xbs.T),b.T,epsilon)):
            p, l, u = linalg.lu(B)
            u = np.asmatrix(linalg.inv(u))
            l = np.asmatrix(linalg.inv(l))
            p = np.asmatrix(linalg.inv(p))
            Eta = []
            i = 0


def etainverse(eta, col):
    ieta = np.matrix(eta)
    dev = eta[col, col]
    ieta[:, col] = -eta[:, col] / dev
    ieta[col, col] = 1 / dev
    return ieta


# returns y*B=Cb
def Btran(p, l, u, Eta, Cb):
    z = Cb
    # Eta.reverse()
    for eta in reversed(Eta):
        z = np.matmul(z, etainverse(eta[0], eta[1]))
    z = np.matmul(z, u)
    z = np.matmul(z, l)
    z = np.matmul(z, p)
    return z


# returns B*d=a
def Ftran(p, l, u, Eta, a):
    z = np.asmatrix(a)
    z = np.matmul(p, z)
    z = np.matmul(l, z)
    z = np.matmul(u, z)
    for eta in Eta:
        z = np.matmul(etainverse(eta[0], eta[1]), z)
    return z.T


def EnteringVariable(An, Cn, Xn, y, rule):
    # rule=0 deantizg, rule=1 blends
    curChoise = -1
    #פונקציית המטרה
    x = Cn - np.matmul(y, An)
    if x.max() < 0:
        return -1

    if rule == 0:
        return np.argmax(x)
    if rule == 1:
        index = np.where(x >0)[1]
        return np.where(Xn[0, :] == Xn[0, index].min())[1][0]
    return curChoise


def LeavingVariable(Xbs, d):
    if d.max() <= 0:
        return -1, -1

    arr =np.divide(Xbs, d)
    l=np.where(arr == np.min(arr[arr >= 0]))[1][0]
    return arr[0,l], l


A = np.matrix([[3, 2, 1, 2], [1, 1, 1, 1], [4, 3, 3, 4]])
b = np.matrix([225., 117., 420.])
c = np.matrix([19, 13, 12, 17])
#A = np.matrix([[1, 0, 0, 1, 0, 0, 1, 0, 0],[0, 1, 0, 0, 1, 0, 0, 1, 0],[0, 0, 1, 0, 0, 1, 0, 0, 1],[0, 0, 0, 1, 1, 1, 0, 0, 0],[0, 0, 0, 0, 0, 0, 1, 1, 1]])
#b = np.matrix([480., 400., 230., 420., 250.])
#c = np.matrix([8, 4, 4, 14, 12, 13, 11, 7, 9])

#A = np.matrix([[0.01, 0.2, 1], [20, 120, 240], [0, 20, 30], [1, 0, 0]])
#b = np.matrix([25., 4000., 1000., 200.])
#c = np.matrix([30, 240, 500])
print(LP(A, b, c, 1))
