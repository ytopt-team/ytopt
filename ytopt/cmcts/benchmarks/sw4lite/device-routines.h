#ifdef SW4_CUDA
void copy_stencilcoefficients( float_sw4* acof, float_sw4* ghcof, float_sw4* bope );
void copy_stencilcoefficients1( float_sw4* acof, float_sw4* ghcof, float_sw4* bope, float_sw4*  );



__global__ void dpdmt_dev( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
			   float_sw4* up, float_sw4* u, float_sw4* um,
			   float_sw4* u2, float_sw4 dt2i, int ghost_points );

__global__ void addsgd4_dev( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
			     float_sw4* a_up, float_sw4* a_u, float_sw4* a_um, float_sw4* a_rho,
			     float_sw4* a_dcx,  float_sw4* a_dcy,  float_sw4* a_dcz,
			     float_sw4* a_strx, float_sw4* a_stry, float_sw4* a_strz,
			     float_sw4* a_cox,  float_sw4* a_coy,  float_sw4* a_coz,
			     float_sw4 beta, int ghost_points );

__global__ void addsgd4_dev_v2( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
			     float_sw4* a_up, float_sw4* a_u, float_sw4* a_um, float_sw4* a_rho,
			     float_sw4* a_dcx,  float_sw4* a_dcy,  float_sw4* a_dcz,
			     float_sw4* a_strx, float_sw4* a_stry, float_sw4* a_strz,
			     float_sw4* a_cox,  float_sw4* a_coy,  float_sw4* a_coz,
			     float_sw4 beta, int ghost_points );

__global__ void addsgd4_dev_rev( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
			     float_sw4* a_up, float_sw4* a_u, float_sw4* a_um, float_sw4* a_rho,
			     float_sw4* a_dcx,  float_sw4* a_dcy,  float_sw4* a_dcz,
			     float_sw4* a_strx, float_sw4* a_stry, float_sw4* a_strz,
			     float_sw4* a_cox,  float_sw4* a_coy,  float_sw4* a_coz,
			     float_sw4 beta, int ghost_points );

__global__ void addsgd4_dev_rev_v2( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
                                    float_sw4* a_up, float_sw4* a_u, float_sw4* a_um, float_sw4* a_rho,
                                    float_sw4* a_dcx,  float_sw4* a_dcy,  float_sw4* a_dcz,
                                    float_sw4* a_strx, float_sw4* a_stry, float_sw4* a_strz,
                                    float_sw4* a_cox,  float_sw4* a_coy,  float_sw4* a_coz,
                                    float_sw4 beta, int ghost_points );

__global__ void rhs4lower_dev( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
			       int nk, float_sw4* a_lu, float_sw4* a_u, float_sw4* a_mu, float_sw4* a_lambda, 
			       float_sw4 h, float_sw4* a_strx, float_sw4* a_stry, float_sw4* a_strz,
			       int ghost_points );

__global__ void rhs4lower_dev_rev( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
				   int nk, float_sw4* a_lu, float_sw4* a_u, float_sw4* a_mu, float_sw4* a_lambda, 
			       float_sw4 h, float_sw4* a_strx, float_sw4* a_stry, float_sw4* a_strz,
			       int ghost_points );

__global__ void addsgd6_dev( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
			     float_sw4* a_up, float_sw4* a_u, float_sw4* a_um, float_sw4* a_rho,
			     float_sw4* a_dcx,  float_sw4* a_dcy,  float_sw4* a_dcz,
			     float_sw4* a_strx, float_sw4* a_stry, float_sw4* a_strz,
			     float_sw4* a_cox,  float_sw4* a_coy,  float_sw4* a_coz,
			     float_sw4 beta, int ghost_points );

__global__ void addsgd6_dev_rev( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
			     float_sw4* a_up, float_sw4* a_u, float_sw4* a_um, float_sw4* a_rho,
			     float_sw4* a_dcx,  float_sw4* a_dcy,  float_sw4* a_dcz,
			     float_sw4* a_strx, float_sw4* a_stry, float_sw4* a_strz,
			     float_sw4* a_cox,  float_sw4* a_coy,  float_sw4* a_coz,
			     float_sw4 beta, int ghost_points );

__global__ void rhs4sgcurv_dev( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
                                       float_sw4* a_u, float_sw4* a_mu, float_sw4* a_lambda, float_sw4* mMetric,
                                       float_sw4* mJ, float_sw4* a_lu, 
                                       int onesided4, float_sw4* a_strx, float_sw4* a_stry, int ghost_points );

__global__ void rhs4sgcurv_dev_rev( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
                                       float_sw4* a_u, float_sw4* a_mu, float_sw4* a_lambda, float_sw4* mMetric,
                                       float_sw4* mJ, float_sw4* a_lu, 
                                       int onesided4, float_sw4* a_strx, float_sw4* a_stry, int ghost_points );

__global__ void rhs4sgcurv_dev_rev_v2( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
				float_sw4* a_u, float_sw4* a_mu, float_sw4* a_lambda, float_sw4* a_met, float_sw4* a_jac, float_sw4* a_lu, 
				int onesided4, float_sw4* a_strx, float_sw4* a_stry, int ghost_points );

__global__ void rhs4sgcurvupper_dev( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
                                       float_sw4* a_u, float_sw4* a_mu, float_sw4* a_lambda, float_sw4* mMetric,
                                       float_sw4* mJ, float_sw4* a_lu, 
                                       float_sw4* a_strx, float_sw4* a_stry, int ghost_points );

__global__ void rhs4sgcurvupper_dev_rev( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
                                       float_sw4* a_u, float_sw4* a_mu, float_sw4* a_lambda, float_sw4* mMetric,
                                       float_sw4* mJ, float_sw4* a_lu, 
                                       float_sw4* a_strx, float_sw4* a_stry, int ghost_points );

__global__ void check_nan_dev( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
			       float_sw4* u, int* retval_dev, int* idx_dev );

__global__ void forcing_dev( float_sw4 t, Sarray* dev_F, int NumberOfGrids, GridPointSource* dev_point_sources,
			     int nptsrc, int* dev_identsources, int nident, bool tt );
__global__ void init_forcing_dev( GridPointSource* point_sources, int nsrc );

__global__ void bcfortsg_dev( int ib, int ie, int jb, int je, int kb, int ke, int* wind,
                              int nx, int ny, int nz, float_sw4* u, float_sw4 h, boundaryConditionType *bccnd,
                              float_sw4* mu, float_sw4* la, float_sw4 t,
                              float_sw4* bforce1, float_sw4* bforce2, float_sw4* bforce3,
                              float_sw4* bforce4, float_sw4* bforce5, float_sw4* bforce6,
                              float_sw4 om, float_sw4 ph, float_sw4 cv,
                              float_sw4* strx, float_sw4* stry );


__global__ void bcfortsg_dev_indrev( int ib, int ie, int jb, int je, int kb, int ke, int* wind,
                                     int nx, int ny, int nz, float_sw4* u, float_sw4 h, boundaryConditionType *bccnd,
                                     float_sw4* mu, float_sw4* la, float_sw4 t,
                                     float_sw4* bforce1, float_sw4* bforce2, float_sw4* bforce3,
                                     float_sw4* bforce4, float_sw4* bforce5, float_sw4* bforce6,
                                     float_sw4 om, float_sw4 ph, float_sw4 cv,
                                     float_sw4* strx, float_sw4* stry );

__global__ void freesurfcurvisg_dev( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
                                     int nz, int side, float_sw4* a_u, float_sw4* a_mu, 
                                     float_sw4* a_la, float_sw4* a_met,
                                     float_sw4* bforce5,float_sw4* a_strx, float_sw4* a_stry, int ghost_points );

__global__ void freesurfcurvisg_dev_rev( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
                                     int nz, int side, float_sw4* a_u, float_sw4* a_mu, 
                                     float_sw4* a_la, float_sw4* a_met,
                                     float_sw4* bforce5,float_sw4* a_strx, float_sw4* a_stry, int ghost_points );

__global__ void enforceCartTopo_dev( int ifirstCart, int ilastCart, int jfirstCart, int jlastCart, int kfirstCart, int klastCart,
                                         int ifirstCurv, int ilastCurv, int jfirstCurv, int jlastCurv, int kfirstCurv, int klastCurv,
                                         float_sw4* a_u1, float_sw4* a_u2, int ghost_points );

__global__ void enforceCartTopo_dev_rev( int ifirstCart, int ilastCart, int jfirstCart, int jlastCart, int kfirstCart, int klastCart,
                                         int ifirstCurv, int ilastCurv, int jfirstCurv, int jlastCurv, int kfirstCurv, int klastCurv,
                                         float_sw4* a_u1, float_sw4* a_u2, int ghost_points );

__global__ void addsgd4c_dev( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
		      float_sw4* a_up, float_sw4* a_u, 
                      float_sw4* a_um, float_sw4* a_rho,
		      float_sw4* a_dcx,  float_sw4* a_dcy, 
                      float_sw4* a_strx, float_sw4* a_stry, 
		      float_sw4* a_jac, float_sw4* a_cox,  float_sw4* a_coy, 
                      float_sw4 beta, int ghost_points );

__global__ void addsgd4c_dev_rev( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
		      float_sw4* a_up, float_sw4* a_u, 
                      float_sw4* a_um, float_sw4* a_rho,
		      float_sw4* a_dcx,  float_sw4* a_dcy, 
                      float_sw4* a_strx, float_sw4* a_stry, 
		      float_sw4* a_jac, float_sw4* a_cox,  float_sw4* a_coy, 
                      float_sw4 beta, int ghost_points );

__global__ void addsgd6c_dev( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
		      float_sw4* a_up, float_sw4* a_u, float_sw4* a_um, float_sw4* a_rho,
		      float_sw4* a_dcx,  float_sw4* a_dcy, 
		      float_sw4* a_strx, float_sw4* a_stry,
		      float_sw4* a_jac, float_sw4* a_cox,  float_sw4* a_coy,  
		      float_sw4 beta, int ghost_points );

__global__ void addsgd6c_dev_rev( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
		      float_sw4* a_up, float_sw4* a_u, float_sw4* a_um, float_sw4* a_rho,
		      float_sw4* a_dcx,  float_sw4* a_dcy, 
		      float_sw4* a_strx, float_sw4* a_stry,
		      float_sw4* a_jac, float_sw4* a_cox,  float_sw4* a_coy,  
		      float_sw4 beta, int ghost_points );

__global__ void pred_dev( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
			  float_sw4* up, float_sw4* u, float_sw4* um, float_sw4* lu, float_sw4* fo,
			  float_sw4* rho, float_sw4 dt2, int ghost_points );

__global__ void pred_dev_rev( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
			  float_sw4* up, float_sw4* u, float_sw4* um, float_sw4* lu, float_sw4* fo,
			  float_sw4* rho, float_sw4 dt2, int ghost_points );

__global__ void corr_dev( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
			  float_sw4* up, float_sw4* lu, float_sw4* fo,
			  float_sw4* rho, float_sw4 dt4, int ghost_points );

__global__ void corr_dev_rev( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
			  float_sw4* up, float_sw4* lu, float_sw4* fo,
			  float_sw4* rho, float_sw4 dt4, int ghost_points );

__global__ void rhs4center_dev( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
				float_sw4* a_lu, float_sw4* a_u, float_sw4* a_mu, float_sw4* a_lambda, 
				float_sw4 h, float_sw4* a_strx, float_sw4* a_stry, float_sw4* a_strz,
				int ghost_points );

__global__ void rhs4center_dev_rev( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
				float_sw4* a_lu, float_sw4* a_u, float_sw4* a_mu, float_sw4* a_lambda, 
				float_sw4 h, float_sw4* a_strx, float_sw4* a_stry, float_sw4* a_strz,
				int ghost_points );

__global__ void rhs4center_dev_v2( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
                                       float_sw4* a_lu, float_sw4* a_u, float_sw4* a_mu, float_sw4* a_lambda,
                                       float_sw4 h, float_sw4* a_strx, float_sw4* a_stry, float_sw4* a_strz,
                                       int ghost_points );

__global__ void rhs4center_dev_rev_v2( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
                                       float_sw4* a_lu, float_sw4* a_u, float_sw4* a_mu, float_sw4* a_lambda,
                                       float_sw4 h, float_sw4* a_strx, float_sw4* a_stry, float_sw4* a_strz,
                                       int ghost_points );

__global__ void rhs4upper_dev( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
			       float_sw4* a_lu, float_sw4* a_u, float_sw4* a_mu, float_sw4* a_lambda, 
			       float_sw4 h, float_sw4* a_strx, float_sw4* a_stry, float_sw4* a_strz,
			       int ghost_points );

__global__ void rhs4upper_dev_rev( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
			       float_sw4* a_lu, float_sw4* a_u, float_sw4* a_mu, float_sw4* a_lambda, 
			       float_sw4 h, float_sw4* a_strx, float_sw4* a_stry, float_sw4* a_strz,
			       int ghost_points );

__global__ void BufferToHaloKernel_dev(float_sw4* block_left, float_sw4* block_right, float_sw4* block_up, float_sw4* block_down,
                        float_sw4 * leftSideEdge, float_sw4 * rightSideEdge, float_sw4 * upSideEdge, float_sw4 * downSideEdge,
                        int ni, int nj, int nk, int m_padding, const int m_neighbor0 ,const int  m_neighbor1, const int m_neighbor2,
                        const int m_neighbor3, const int mpi_process_null_cuda);

__global__ void BufferToHaloKernel_dev_rev(float_sw4* block_left, float_sw4* block_right, float_sw4* block_up, float_sw4* block_down,
                        float_sw4 * leftSideEdge, float_sw4 * rightSideEdge, float_sw4 * upSideEdge, float_sw4 * downSideEdge,
                        int ni, int nj, int nk, int m_padding, const int m_neighbor0 ,const int  m_neighbor1, const int m_neighbor2,
                        const int m_neighbor3, const int mpi_process_null_cuda);

__global__ void BufferToHaloKernel_dev_rev_v2(float_sw4* block_left, float_sw4* block_right,
                float_sw4 * leftSideEdge, float_sw4 * rightSideEdge,
                int ni, int nj, int nk, int m_padding, int size, int nstep, const int m_neighbor_left ,const int  m_neighbor_right, const int mpi_process_null );

__global__ void HaloToBufferKernel_dev(float_sw4* block_left, float_sw4* block_right, float_sw4* block_up, float_sw4* block_down,
                        float_sw4 * leftSideEdge, float_sw4 * rightSideEdge, float_sw4 * upSideEdge, float_sw4 * downSideEdge,
                        int ni, int nj, int nk, int m_padding, const int m_neighbor0 ,const int  m_neighbor1, const int m_neighbor2,
                        const int m_neighbor3, const int mpi_process_null_cuda);

__global__ void HaloToBufferKernel_dev_rev(float_sw4* block_left, float_sw4* block_right, float_sw4* block_up, float_sw4* block_down,
                        float_sw4 * leftSideEdge, float_sw4 * rightSideEdge, float_sw4 * upSideEdge, float_sw4 * downSideEdge,
                        int ni, int nj, int nk, int m_padding, const int m_neighbor0 ,const int  m_neighbor1, const int m_neighbor2,
                        const int m_neighbor3, const int mpi_process_null_cuda);

__global__ void HaloToBufferKernel_dev_rev_v2(float_sw4* block_left, float_sw4* block_right,
                        float_sw4 * leftSideEdge, float_sw4 * rightSideEdge,
                        int ni, int nj, int nk, int m_padding, int size, int nstep, const int m_neighbor_left ,const int  m_neighbor_right, const int mpi_process_null);










// *****************************************************************************
// *****************************************************************************
// *****************************************************************************
// New GPU kernels from G. Thomas-Collignon

__global__ void HaloToBufferKernelX_dev(float_sw4* block_left, float_sw4* block_right,
                                        float_sw4 * leftSideEdge, float_sw4 * rightSideEdge,
                                        int ni, int nj, int nk, int m_padding, const int m_neighbor2,
                                        const int m_neighbor3, const int mpi_process_null);

__global__ void HaloToBufferKernelX_dev_rev(float_sw4* block_left, float_sw4* block_right,
                                            float_sw4 * leftSideEdge, float_sw4 * rightSideEdge,
                                            int ni, int nj, int nk, int m_padding, const int m_neighbor2,
                                            const int m_neighbor3, const int mpi_process_null);

__global__ void HaloToBufferKernelY_dev(float_sw4* block_up, float_sw4* block_down,
                                        float_sw4 * upSideEdge, float_sw4 * downSideEdge,
                                        int ni, int nj, int nk, int m_padding, const int m_neighbor0,
                                        const int  m_neighbor1, const int mpi_process_null);

__global__ void HaloToBufferKernelY_dev_rev(float_sw4* block_up, float_sw4* block_down,
                                            float_sw4 * upSideEdge, float_sw4 * downSideEdge,
                                            int ni, int nj, int nk, int m_padding, const int m_neighbor0,
                                            const int  m_neighbor1, const int mpi_process_null);

__global__ void BufferToHaloKernelX_dev(float_sw4* block_left, float_sw4* block_right, float_sw4 * leftSideEdge, 
                                        float_sw4 * rightSideEdge, int ni, int nj, int nk, int m_padding, 
                                        const int m_neighbor2, const int m_neighbor3, const int mpi_process_null );

__global__ void BufferToHaloKernelX_dev_rev(float_sw4* block_left, float_sw4* block_right,
                                            float_sw4 * leftSideEdge, float_sw4 * rightSideEdge, 
                                            int ni, int nj, int nk, int m_padding, const int m_neighbor2,
                                            const int m_neighbor3, const int mpi_process_null);

__global__ void BufferToHaloKernelY_dev(float_sw4* block_up, float_sw4* block_down,
                                        float_sw4* upSideEdge, float_sw4* downSideEdge,
                                        int ni, int nj, int nk, int m_padding, const int m_neighbor0 ,
                                        const int  m_neighbor1, const int mpi_process_null );

__global__ void BufferToHaloKernelY_dev_rev(float_sw4* block_up, float_sw4* block_down,
                                            float_sw4* upSideEdge, float_sw4* downSideEdge,
                                            int ni, int nj, int nk, int m_padding, const int m_neighbor0,
                                            const int m_neighbor1, const int mpi_process_null );



// *****************************************************************************

void rhs4_pred_gpu (int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
                    int ni, int nj, int nk,
                    float_sw4* a_up, float_sw4* a_u, float_sw4* a_um,
                    float_sw4* a_mu, float_sw4* a_lambda, float_sw4* a_rho, float_sw4* a_fo,
                    float_sw4* a_strx, float_sw4* a_stry, float_sw4* a_strz, 
                    float_sw4 h, float_sw4 dt, bool c_order, cudaStream_t stream);

void rhs4_corr_gpu (int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
                    int ni, int nj, int nk,
                    float_sw4* a_up, float_sw4* a_u,
                    float_sw4* a_mu, float_sw4* a_lambda, float_sw4* a_rho, float_sw4* a_fo,
                    float_sw4* a_strx, float_sw4* a_stry, float_sw4* a_strz, 
                    float_sw4 h, float_sw4 dt, bool c_order, cudaStream_t stream);

void rhs4_X_pred_gpu (int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
		      int ni, int nj, int nk,
		      float_sw4* a_up, float_sw4* a_u, float_sw4* a_um,
		      float_sw4* a_mu, float_sw4* a_lambda, float_sw4* a_rho, float_sw4* a_fo, 
		      float_sw4* a_strx, float_sw4* a_stry, float_sw4* a_strz, 
		      float_sw4 h, float_sw4 dt, bool c_order, cudaStream_t stream);

void rhs4_X_corr_gpu (int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
		      int ni, int nj, int nk,
		      float_sw4* a_up, float_sw4* a_u,
		      float_sw4* a_mu, float_sw4* a_lambda, float_sw4* a_rho, float_sw4* a_fo, 
		      float_sw4* a_strx, float_sw4* a_stry, float_sw4* a_strz, 
		      float_sw4 h, float_sw4 dt, bool c_order, cudaStream_t stream);

void rhs4_Y_pred_gpu (int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
		      int ni, int nj, int nk,
		      float_sw4* a_up, float_sw4* a_u, float_sw4* a_um,
		      float_sw4* a_mu, float_sw4* a_lambda, float_sw4* a_rho, float_sw4* a_fo,
		      float_sw4* a_strx, float_sw4* a_stry, float_sw4* a_strz, 
		      float_sw4 h, float_sw4 dt, bool c_order, cudaStream_t stream);

void rhs4_Y_corr_gpu (int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
		      int ni, int nj, int nk,
		      float_sw4* a_up, float_sw4* a_u,
		      float_sw4* a_mu, float_sw4* a_lambda, float_sw4* a_rho, float_sw4* a_fo,
		      float_sw4* a_strx, float_sw4* a_stry, float_sw4* a_strz, 
		      float_sw4 h, float_sw4 dt, bool c_order, cudaStream_t stream);

void rhs4_lowk_pred_gpu (int ifirst, int ilast, int jfirst, int jlast,
			 int ni, int nj, int nk, int nz,
			 float_sw4* a_up, float_sw4* a_u, float_sw4* a_um,
			 float_sw4* a_mu, float_sw4* a_lambda, float_sw4* a_rho, float_sw4* a_fo,
			 float_sw4* a_strx, float_sw4* a_stry, float_sw4* a_strz,
			 float_sw4 h, float_sw4 dt, bool c_order, cudaStream_t stream);

void rhs4_highk_pred_gpu (int ifirst, int ilast, int jfirst, int jlast,
			 int ni, int nj, int nk, int nz,
			 float_sw4* a_up, float_sw4* a_u, float_sw4* a_um,
			 float_sw4* a_mu, float_sw4* a_lambda, float_sw4* a_rho, float_sw4* a_fo,
			 float_sw4* a_strx, float_sw4* a_stry, float_sw4* a_strz,
			 float_sw4 h, float_sw4 dt, bool c_order, cudaStream_t stream);

void rhs4_lowk_corr_gpu (int ifirst, int ilast, int jfirst, int jlast,
			 int ni, int nj, int nk, int nz,
			 float_sw4* a_up, float_sw4* a_u,
			 float_sw4* a_mu, float_sw4* a_lambda, float_sw4* a_rho, float_sw4* a_fo,
			 float_sw4* a_strx, float_sw4* a_stry, float_sw4* a_strz,
			 float_sw4 h, float_sw4 dt, bool c_order, cudaStream_t stream);

void rhs4_highk_corr_gpu (int ifirst, int ilast, int jfirst, int jlast,
			 int ni, int nj, int nk, int nz,
			 float_sw4* a_up, float_sw4* a_u,
			 float_sw4* a_mu, float_sw4* a_lambda, float_sw4* a_rho, float_sw4* a_fo,
			 float_sw4* a_strx, float_sw4* a_stry, float_sw4* a_strz,
			 float_sw4 h, float_sw4 dt, bool c_order, cudaStream_t stream);

void addsgd4_gpu (int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
		  int ni, int nj, int nk,
		  float_sw4* a_up, float_sw4* a_u, float_sw4* a_um, float_sw4* a_rho,
		  float_sw4* a_dcx,  float_sw4* a_dcy,  float_sw4* a_dcz,
		  float_sw4* a_strx, float_sw4* a_stry, float_sw4* a_strz,
		  float_sw4* a_cox,  float_sw4* a_coy,  float_sw4* a_coz,
		  float_sw4 beta, int c_order, cudaStream_t stream);

void addsgd4_X_gpu (int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
		    int ni, int nj, int nk,
		    float_sw4* a_up, float_sw4* a_u, float_sw4* a_um, float_sw4* a_rho,
		    float_sw4* a_dcx,  float_sw4* a_dcy,  float_sw4* a_dcz,
		    float_sw4* a_strx, float_sw4* a_stry, float_sw4* a_strz,
		    float_sw4* a_cox,  float_sw4* a_coy,  float_sw4* a_coz,
		    float_sw4 beta, int c_order, cudaStream_t stream);

void addsgd4_Y_gpu (int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
		    int ni, int nj, int nk,
		    float_sw4* a_up, float_sw4* a_u, float_sw4* a_um, float_sw4* a_rho,
		    float_sw4* a_dcx,  float_sw4* a_dcy,  float_sw4* a_dcz,
		    float_sw4* a_strx, float_sw4* a_stry, float_sw4* a_strz,
		    float_sw4* a_cox,  float_sw4* a_coy,  float_sw4* a_coz,
		    float_sw4 beta, int c_order, cudaStream_t stream);

void bcfortsg_gpu (int ib, int ie, int jb, int je, int kb, int ke, int* wind,
		   int nx, int ny, int nz, float_sw4* a_u, float_sw4 h, boundaryConditionType *bccnd,
		   float_sw4* mu, float_sw4* la, float_sw4 t,
		   float_sw4* bforce1, float_sw4* bforce2, float_sw4* bforce3,
		   float_sw4* bforce4, float_sw4* bforce5, float_sw4* bforce6,
		   float_sw4 om, float_sw4 ph, float_sw4 cv,
		   float_sw4* strx, float_sw4* stry, int c_order, cudaStream_t stream);

__global__ void extractRecordData_dev( int nt, int* mode, int* i0v, int* j0v, int* k0v,
				       int* g0v, float_sw4** urec, Sarray* Um2, Sarray* U,
				       float_sw4 dt, float_sw4* h, int numberOfCartesianGrids, 
				       Sarray* mMetric, Sarray* mJ );

#endif
