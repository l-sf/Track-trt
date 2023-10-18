
#ifndef TRT_TENSOR_HPP
#define TRT_TENSOR_HPP

#include <string>
#include <memory>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>

struct CUstream_st;
typedef CUstream_st CUStreamRaw;

#define CURRENT_DEVICE_ID 0

namespace TRT {

    typedef CUStreamRaw* CUStream;

    enum class DataHead : int{
        Init   = 0,
        Device = 1,
        Host   = 2
    };

    
    /// -----------------------------------------------------------------------------------------------
    /// ----------------------------------- MixMemory类 -----------------------------------------------
    /// --------------- 对 memory 进行封装，使得内存分配复制自动管理，避免手动管理的繁琐 ----------------------
    /// -----------------------------------------------------------------------------------------------
    
    class MixMemory {
    public:
        explicit MixMemory(int device_id = CURRENT_DEVICE_ID);
        MixMemory(void* cpu, size_t cpu_size, void* gpu, size_t gpu_size);
        virtual ~MixMemory();
        /// 以下四个函数用来 在GPU或CPU上分配size大小的内存，并返回对应的内存地址
        void* gpu(size_t size);
        void* cpu(size_t size);
        void* gpu() const { return gpu_; }
        void* cpu() const { return cpu_; }

        /// 获取内存空间大小、device_id
        size_t gpu_size() const { return gpu_size_;  }
        size_t cpu_size() const { return cpu_size_;  }
        int device_id()   const { return device_id_; }

        /// 是否属于我自己分配的gpu/cpu
        bool owner_gpu()  const { return owner_gpu_; }
        bool owner_cpu()  const { return owner_cpu_; }

        /// 以下三个函数为释放内存空间
        void release_gpu();
        void release_cpu();
        void release_all();

        /// 刷新内存（重新初始化）
        void reference_data(void* cpu, size_t cpu_size, void* gpu, size_t gpu_size);

    private:
        void* cpu_ = nullptr;
        size_t cpu_size_ = 0;
        bool owner_cpu_ = true;
        int device_id_ = 0;

        void* gpu_ = nullptr;
        size_t gpu_size_ = 0;
        bool owner_gpu_ = true;
    };


    /// -----------------------------------------------------------------------------------------------
    /// -------------------------------------- Tensor类 ------------------------------------------------
    /// ------------- 对tensor进行封装，张量是CNN中常见的基本单元，尤其是计算偏移量的工作需要封装； -------------
    /// ------------- 其次是内存的复制、分配需要引用memory进行包装，避免使用时面对指针不好管控。 ----------------
    /// ------------------------------------------------------------------------------------------------
    
    class Tensor {
    public:
        Tensor(const Tensor& other) = delete;
        Tensor& operator = (const Tensor& other) = delete;

        explicit Tensor(std::unique_ptr<MixMemory> data = nullptr, int device_id = CURRENT_DEVICE_ID);
        explicit Tensor(int n, int c, int h, int w, std::unique_ptr<MixMemory> data = nullptr, int device_id = CURRENT_DEVICE_ID);
        explicit Tensor(int ndims, const int* dims, std::unique_ptr<MixMemory> data = nullptr, int device_id = CURRENT_DEVICE_ID);
        explicit Tensor(const std::vector<int>& dims, std::unique_ptr<MixMemory> data = nullptr, int device_id = CURRENT_DEVICE_ID);
        virtual ~Tensor();

        /// 以下函数用来 让外部获取 Tensor 的一些基本属性
        int numel() const;
        int ndims() const{return shape_.size();}
        int size(int index)  const{return shape_[index];}
        int shape(int index) const{return shape_[index];}
        int batch()   const{return shape_[0];}
        int channel() const{return shape_[1];}
        int height()  const{return shape_[2];}
        int width()   const{return shape_[3];}

        const std::vector<int>& dims() const { return shape_; }
        const std::vector<size_t>& strides() const {return strides_;}
        int bytes()                    const { return bytes_; }
        int bytes(int start_axis)      const { return count(start_axis) * sizeof(float); }
        DataHead head()                const { return head_; }
        std::shared_ptr<Tensor> clone() const;
        Tensor& release();
        bool empty() const;
        int  count(int start_axis = 0) const;
        int device() const{return device_id_;}
        const char* shape_string() const{return shape_string_;}
        const char* descriptor() const;

        /// 以下函数用来 将 Tensor 拷贝到 GPU 或 CPU 上，并返回对应内存地址
        Tensor& to_gpu(bool copy=true);
        Tensor& to_cpu(bool copy=true);
        void* cpu() const { ((Tensor*)this)->to_cpu(); return data_->cpu(); }
        void* gpu() const { ((Tensor*)this)->to_gpu(); return data_->gpu(); }
        template<class DType> DType* gpu() { return (DType*)gpu(); }
        template<class DType> DType* cpu() { return (DType*)cpu(); }

        template<typename DType, typename ... Args>
        DType* cpu(int i, Args&& ... args) { return cpu<DType>() + offset(i, args...); }
        template<typename DType, typename ... Args>
        DType* gpu(int i, Args&& ... args) { return gpu<DType>() + offset(i, args...); }
        template<typename DType, typename ... Args>
        DType& at(int i, Args&& ... args) { return *(cpu<DType>() + offset(i, args...)); }

        Tensor& copy_from_gpu(size_t offset, const void* src, size_t num_element, int device_id = CURRENT_DEVICE_ID);
        Tensor& copy_from_cpu(size_t offset, const void* src, size_t num_element);

        /// 以下函数用来 对Tensor做一些操作
        template<typename ... Args>
        Tensor& resize(int dim_size, Args ... dim_size_args){
            const int dim_size_array[] = {dim_size, dim_size_args...};
            return resize(sizeof...(dim_size_args) + 1, dim_size_array);
        }
        Tensor& resize(int ndims, const int* dims);
        Tensor& resize(const std::vector<int>& dims);
        Tensor& resize_single_dim(int idim, int size);

        template<typename ... Args>
        int offset(int index, Args ... index_args) const{
            const int index_array[] = {index, index_args...};
            return offset_array(sizeof...(index_args) + 1, index_array);
        }
        int offset_array(const std::vector<int>& index) const;
        int offset_array(size_t size, const int* index_array) const;

        std::shared_ptr<MixMemory> get_workspace() const { return workspace_; }
        Tensor& set_workspace(std::shared_ptr<MixMemory> workspace) {workspace_ = workspace; return *this;}
        Tensor& synchronize();
        bool is_stream_owner() const {return stream_owner_;}
        CUStream get_stream() const{return stream_;}
        Tensor& set_stream(CUStream stream, bool owner=false){stream_ = stream; stream_owner_ = owner; return *this;}

        Tensor& set_mat     (int n, const cv::Mat& image);
        Tensor& set_norm_mat_invert(int n, const cv::Mat& image, float mean[3], float std[3]);
        Tensor& set_norm_mat(int n, const cv::Mat& image, float mean[3], float std[3]);
        cv::Mat at_mat(int n = 0, int c = 0) { return {height(), width(), CV_32F, cpu<float>(n, c)}; }

        void reference_data(const std::vector<int>& shape, void* cpu_data, size_t cpu_size, void* gpu_data, size_t gpu_size);

    private:
        Tensor& compute_shape_string();
        Tensor& adajust_memory_by_update_dims();  // 设置bytes大小
        void setup_data(std::unique_ptr<MixMemory> data);

    private:
        std::vector<int> shape_;
        std::vector<size_t> strides_;  // 记录当前维度上index改变1,内存上应该改变多少位
        size_t bytes_    = 0;
        DataHead head_   = DataHead::Init;
        CUStream stream_ = nullptr;
        bool stream_owner_ = false;
        int device_id_   = 0;
        char shape_string_[100]{};  // 存放形状字符串描述，如："1 x 3 x 640 x 640"
        char descriptor_string_[100]{};
        std::unique_ptr<MixMemory> data_;
        std::shared_ptr<MixMemory> workspace_;  // 存放附带的一些内存，如预处理的matrix、image要放到GPU上
    };
};

#endif // TRT_TENSOR_HPP