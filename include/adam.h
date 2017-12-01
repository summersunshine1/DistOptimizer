#include<vector>
#include<math.h>
#include <iostream>

namespace distlr {
    
class Adam {
    public:
        explicit Adam(int gradsize, int per_grad_dim = 1, float alpha = 0.01, float beta1 = 0.9, float beta2 = 0.999,
        float epsilo = 1e-5): grad_size(gradsize), per_grad_dim(per_grad_dim), alpha(alpha), beta1(beta1), beta2(beta2),epsilo(epsilo)
        {
            m.resize(gradsize);
            v.resize(gradsize);
            for (size_t i = 0; i < m.size(); ++i) {
                m[i] = 0;
                v[i] = 0;
            }
        }
        Adam(){
        }
        ~Adam() {
        }
        
        float getgrad(float grad,int i, int iter)
        {   
            float mhat = 0;
            float vhat = 0;
            m[i] = beta1*m[i]+(1-beta1)*grad;
            v[i] = beta2*v[i]+(1-beta2)*grad*grad;
            mhat = m[i]/(1-pow(beta1,iter+1));
            vhat = v[i]/(1-pow(beta2,iter+1));
            float updated_grad = alpha*mhat/(sqrt(vhat)+epsilo);
            // std::cout<<grad<<" "<<updated_grad<<std::endl;
            return updated_grad;
        }
    private:
        float alpha;
        float beta1;
        float beta2;
        float epsilo;
        std::vector<float> m;
        std::vector<float> v;
        int grad_size;
        int per_grad_dim; 
};
}