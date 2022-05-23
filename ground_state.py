import subprocess
import os
import sys
import time
import shutil
import math
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg

import parameters as pam
import hamiltonian as ham
import lattice as lat
import variational_space as vs 
# import utility as util

def get_ground_state(matrix, VS, S_val,Sz_val):  
    '''
    Obtain the ground state info, namely the lowest peak in Aw_dd's component
    in particular how much weight of various d8 channels: a1^2, b1^2, b2^2, e^2
    '''        
    print ('start getting ground state')
#     # in case eigsh does not work but matrix is actually small, e.g. Mc=1 (CuO4)
#     M_dense = matrix.todense()
#     print ('H=')
#     print (M_dense)
    
#     for ii in range(0,1325):
#         for jj in range(0,1325):
#             if M_dense[ii,jj]>0 and ii!=jj:
#                 print ii,jj,M_dense[ii,jj]
#             if M_dense[ii,jj]==0 and ii==jj:
#                 print ii,jj,M_dense[ii,jj]
                    
                
#     vals, vecs = np.linalg.eigh(M_dense)
#     vals.sort()                                                               #calculate atom limit
#     print ('lowest eigenvalue of H from np.linalg.eigh = ')
#     print (vals)
    
    # in case eigsh works:
    Neval = pam.Neval
    vals, vecs = sps.linalg.eigsh(matrix, k=Neval, which='SA')                 #normal calculate
    vals.sort()
    print ('lowest eigenvalue of H from np.linalg.eigsh = ')
    print (vals)
    
    # get state components in GS and another 9 higher states; note that indices is a tuple
    for k in range(0,1):
        #if vals[k]<pam.w_start or vals[k]>pam.w_stop:
        #if vals[k]<11.5 or vals[k]>14.5:
        #if k<Neval:
        #    continue
            
        print ('eigenvalue = ', vals[k])
        indices = np.nonzero(abs(vecs[:,k])>0.05)
        wgt_d8 = np.zeros(6)
        wgt_d9L = np.zeros(6)
        wgt_d10L2 = np.zeros(2)
        
        m = 0
        sumweight=0
        weight1=0
        weight2=0
        weight3=0
        weight4=0
        weight5=0
        weight6=0
        weight7=0
        weight8=0
        weight9=0
        weight10=0
        weight11=0
        weight12=0
        weight13=0
        weight14=0
        
        print ("Compute the weights in GS (lowest Aw peak)")
        #for i in indices[0]:
        for i in range(0,len(vecs[:,k])):
            # state is original state but its orbital info remains after basis change
            state = VS.get_state(VS.lookup_tbl[i])
            
            s1 = state['hole1_spin']
            s2 = state['hole2_spin']
            orb1 = state['hole1_orb']
            orb2 = state['hole2_orb']
            x1, y1, z1 = state['hole1_coord']
            x2, y2, z2 = state['hole2_coord']

            # also obtain the total S and Sz of the state
            S12  = S_val[i]
            Sz12 = Sz_val[i]

            o12 = sorted([orb1,orb2])
            o12 = tuple(o12)

            if i in indices[0]:
                print (' state ', orb1,x1,y1,z1,orb2,x2,y2,z2, 'S=',S12,'Sz=',Sz12, \
                  ", weight = ", abs(vecs[i,k])**2)

            if (o12[0]=='px1' and o12[1]=='px1') or (o12[0]=='py2' and o12[1]=='py2')  and z1==z2==0 and S12==0 and Sz12==0:
                 weight1+=abs(vecs[i,k])**2
            if o12[0]=='px1' and o12[1]=='py2' and z1==z2==0 and S12==0 and Sz12==0:
                 weight2+=abs(vecs[i,k])**2
                    


            if o12[0]=='dx2y2' and o12[1]=='dx2y2' and z1==z2==0 and S12==0 and Sz12==0:
                 weight3+=abs(vecs[i,k])**2
            if o12[0]=='dx2y2' and o12[1]=='dx2y2' and z1==z2==1 and S12==0 and Sz12==0:
                 weight4+=abs(vecs[i,k])**2
            if o12[0]=='dx2y2' and o12[1]=='dx2y2' and z1==1 and z2==0 and S12==0 and Sz12==0:
                 weight5+=abs(vecs[i,k])**2   
            if o12[0]=='d3z2r2' and o12[1]=='dx2y2' and z1==z2==0 and S12==1 and Sz12==0:
                 weight6+=abs(vecs[i,k])**2
            if o12[0]=='d3z2r2' and o12[1]=='dx2y2' and z1==z2==1 and S12==1 and Sz12==0:
                 weight7+=abs(vecs[i,k])**2
            if o12[0]=='d3z2r2' and o12[1]=='dx2y2' and z1==1 and z2==0 and S12==0 and Sz12==0:
                 weight8+=abs(vecs[i,k])**2  
            
                    
            if o12[0]=='dx2y2' and (o12[1]=='px1' or o12[1]=='py2') and z1==z2==0 and S12==0 and Sz12==0:
                 weight9+=abs(vecs[i,k])**2
            if o12[0]=='dx2y2' and (o12[1]=='px1' or o12[1]=='py2') and z1==z2==1 and S12==0 and Sz12==0:
                 weight10+=abs(vecs[i,k])**2 
            if o12[0]=='dx2y2' and (o12[1]=='px1' or o12[1]=='py2') and ((z1==1 and z2==0) or (z1==0 and z2==1)) and S12==0 and Sz12==0:
                 weight11+=abs(vecs[i,k])**2

            if o12[0]=='d3z2r2' and (o12[1]=='px1' or o12[1]=='py2') and z1==z2==0 and S12==0 and Sz12==0:
                 weight12+=abs(vecs[i,k])**2
            if o12[0]=='d3z2r2' and (o12[1]=='px1' or o12[1]=='py2') and z1==z2==1 and S12==0 and Sz12==0:
                 weight13+=abs(vecs[i,k])**2 
            if o12[0]=='d3z2r2' and (o12[1]=='px1' or o12[1]=='py2') and ((z1==1 and z2==0) or (z1==0 and z2==1)) and S12==0 and Sz12==0:
                 weight14+=abs(vecs[i,k])**2


  
                    
                    
#             if (o12[0]=='px' and o12[1]=='px') or (o12[0]=='py' and o12[1]=='py') and z1==1 and z2==1 and S12==0 and Sz12==0:
#                   weight1+=abs(vecs[i,k])**2   
#             if o12[0]=='px' and o12[1]=='py' and z1==1 and z2==1 and S12==0 and Sz12==0:
#                   weight2+=abs(vecs[i,k])**2   
#             if o12[0]=='dx2y2' and (o12[1]=='px' or o12[1]=='py')  and z1==1 and z2==1 and S12==0 and Sz12==0:
#                   weight3+=abs(vecs[i,k])**2                     
#             if o12[0]=='d3z2r2' and (o12[1]=='px' or o12[1]=='py')  and z1==1 and z2==1 and S12==0 and Sz12==0:
#                   weight4+=abs(vecs[i,k])**2 
#             if o12[0]=='d3z2r2' and o12[1]=='d3z2r2' and z1==1 and z2==1 and S12==0 and Sz12==0:
#                   weight5+=abs(vecs[i,k])**2   
#             if o12[0]=='d3z2r2' and o12[1]=='dx2y2' and z1==1 and z2==1 and S12==1 and Sz12==0:
#                   weight6+=abs(vecs[i,k])**2                       
#             if o12[0]=='dx2y2' and o12[1]=='dx2y2' and z1==1 and z2==1 and S12==0 and Sz12==0:
#                   weight7+=abs(vecs[i,k])**2                       
                    
            sumweight=sumweight+abs(vecs[i,k])**2
        print ('sumweight=',sumweight)


# #             # record the weights of 1A1 and 3B1 states a1a1, b1b1, ..., a1b1 in G.S.
# #             if o12[0]==o12[1]=='d3z2r2':
# #                 wgt_d8[0] += abs(vecs[i,k])**2
# #             if o12==('dx2y2','dx2y2'):
# #                 wgt_d8[1] += abs(vecs[i,k])**2
# #             if o12[0]==o12[1]=='dxy':
# #                 wgt_d8[2] += abs(vecs[i,k])**2
# # #             if AorB_sym[i]==1:
# # #                 wgt_d8[3] += abs(vecs[i,k])**2
# #             if o12[0]=='d3z2r2' and o12[1]=='dx2y2':
# #                 wgt_d8[4] += abs(vecs[i,k])**2
# #             if o12[0]=='d3z2r2' and o12[1]=='dxy':
# #                 wgt_d8[5] += abs(vecs[i,k])**2




#         txt=open('px1px1orpy2py210','a')                                  #tz=0.1-2,步长0.1
#         txt.write(str(weight1)+'\n')
#         txt.close()
#         txt=open('px1py210','a')                                  #tz=0.1-2,步长0.1
#         txt.write(str(weight2)+'\n')
#         txt.close()




#         txt=open('dx2y2dx2y2z=010','a')                                  #tz=0.1-2,步长0.1
#         txt.write(str(weight3)+'\n')
#         txt.close()
#         txt=open('dx2y2dx2y2z=110','a')                                  #tz=0.1-2,步长0.1
#         txt.write(str(weight4)+'\n')
#         txt.close()
#         txt=open('dx2y2dx2y2z=1z=010','a')                                  #tz=0.1-2,步长0.1
#         txt.write(str(weight5)+'\n')
#         txt.close()     
#         txt=open('d3z2r2dx2y2z=010','a')                                  #tz=0.1-2,步长0.1
#         txt.write(str(weight6)+'\n')
#         txt.close()
#         txt=open('d3z2r2dx2y2z=110','a')                                  #tz=0.1-2,步长0.1
#         txt.write(str(weight7)+'\n')
#         txt.close()
#         txt=open('d3z2r2dx2y2z=1010','a')                                  #tz=0.1-2,步长0.1
#         txt.write(str(weight8)+'\n')
#         txt.close()  


#         txt=open('dx2y2pz=0z=010','a')                                  #tz=0.1-2,步长0.1
#         txt.write(str(weight9)+'\n')
#         txt.close()
#         txt=open('dx2y2pz=1z=110','a')                                  #tz=0.1-2,步长0.1
#         txt.write(str(weight10)+'\n')
#         txt.close()
#         txt=open('dx2y2pz=1z=010','a')                                  #tz=0.1-2,步长0.1
#         txt.write(str(weight11)+'\n')
#         txt.close()
#         txt=open('d3z2r2pz=0z=010','a')                                  #tz=0.1-2,步长0.1
#         txt.write(str(weight12)+'\n')
#         txt.close()
#         txt=open('d3z2r2pz=1z=110','a')                                  #tz=0.1-2,步长0.1
#         txt.write(str(weight13)+'\n')
#         txt.close()
#         txt=open('d3z2r2pz=1z=010','a')                                  #tz=0.1-2,步长0.1
#         txt.write(str(weight14)+'\n')
#         txt.close()


#         txt=open('eigenvalue10','a')                                  #tz=0.1-2,步长0.1
#         txt.write(str(vals[k])+'\n')
#         txt.close()



#         txt=open('pxpx5','a')                                  #tz=0.1-2,步长0.1
#         txt.write(str(weight1)+'\n')
#         txt.close()
#         txt=open('pxpy5','a')                                  #tz=0.1-2,步长0.1
#         txt.write(str(weight2)+'\n')
#         txt.close()
#         txt=open('dx2y2px5','a')                                  #tz=0.1-2,步长0.1
#         txt.write(str(weight3)+'\n')
#         txt.close()
#         txt=open('d3z2r2py5','a')                                  #tz=0.1-2,步长0.1
#         txt.write(str(weight4)+'\n')
#         txt.close()
#         txt=open('d3z2r2d3z2r25','a')                                  #tz=0.1-2,步长0.1
#         txt.write(str(weight5)+'\n')
#         txt.close()
#         txt=open('d3z2r2dx2y25','a')                                  #tz=0.1-2,步长0.1
#         txt.write(str(weight6)+'\n')
#         txt.close()
#         txt=open('dx2y2dx2y25','a')                                  #tz=0.1-2,步长0.1
#         txt.write(str(weight7)+'\n')
#         txt.close()
#         weight8=weight1+weight2
#         txt=open('d10L2','a')                                  #tz=0.1-2,步长0.1
#         txt.write(str(weight8)+'\n')
#         txt.close()



    return vals, vecs, wgt_d8, wgt_d9L, wgt_d10L2
