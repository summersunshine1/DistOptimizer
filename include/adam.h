#include<vector>
#include<math.h>
#include <iostream>
    
class Adam {
    public:
        explicit Adam(int gradsize, int per_grad_dim = 1, float alpha = 0.01, float beta1 = 0.9, float beta2 = 0.999,
        float epsilo = 1e-8): grad_size(gradsize), per_grad_dim(per_grad_dim), alpha(alpha), beta1(beta1), beta2(beta2),epsilo(epsilo)
        {
            m.resize(gradsize);
            v.resize(gradsize);
            for (size_t i = 0; i < m.size(); ++i) {
                m[i] = 0.0;
                v[i] = 0.0;
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
            v[i] = beta2*v[i]+(1-beta2)*pow(grad,2);
            mhat = m[i]/(1-pow(beta1,iter));
            vhat = v[i]/(1-pow(beta2,iter));
            float updated_grad = alpha*mhat/(sqrt(vhat)+epsilo);
            // std::cout<<grad<<" "<<updated_grad<<std::endl;
            return updated_grad;
            
        }
        
        float getmaxgrad(float grad,int i, int iter)
        {
            m[i] = beta1*m[i]+(1-beta1)*grad;
            v[i] = std::max(beta2*v[i],std::abs(grad));
            float updates_grad = alpha/(1-pow(beta1,iter))*m[i]/(v[i]+epsilo);
            return updates_grad;
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

class Adagrad {
    public:
        explicit Adagrad(int gradsize,float alpha = 0.001,float epsilo = 1e-10): 
        grad_size(gradsize), alpha(alpha), epsilo(epsilo)
        {
            n.resize(gradsize);
        }
        Adagrad(){
        }
        ~Adagrad() {
        }
        
        float getgrad(float grad,int i, int iter)
        {   
            n[i] += grad*grad;
            float updated_grad = alpha*grad/sqrt(n[i]+epsilo);
            // std::cout<<grad<<" "<<updated_grad<<std::endl;
            return updated_grad;
        }
    private:
        float alpha;
        float epsilo;
        std::vector<float> n;
        int grad_size;
};