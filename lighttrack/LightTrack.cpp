
#include "LightTrack.hpp"
#include <queue>
#include <condition_variable>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/cuda.hpp>
#include "trt_common/trt_infer.hpp"
#include "trt_common/ilogger.hpp"
#include "trt_common/preprocess_kernel.cuh"
#include "trt_common/cuda_tools.hpp"

namespace LightTrack {

    using namespace std;

    void decode_kernel_invoker(
            float* predict, int num_bboxes, int num_classes, float confidence_threshold,
            int scale_expand, float* invert_affine_matrix, float* parray,
            int max_objects, cudaStream_t stream
    );

    struct AffineMatrix{
        float i2d[6];       // image to dst(network), 2x3 matrix
        float d2i[6];       // dst to image, 2x3 matrix

        void compute(const cv::Size& from, const cv::Size& to){
            float scale_x = to.width / (float)from.width;
            float scale_y = to.height / (float)from.height;
            float scale = std::min(scale_x, scale_y);

            i2d[0] = scale;  i2d[1] = 0;  i2d[2] = (-scale * from.width + to.width + scale - 1) * 0.5f;
            i2d[3] = 0;  i2d[4] = scale;  i2d[5] = (-scale * from.height + to.height + scale - 1) * 0.5f;

            cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);
            cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);
            cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);
        }

        cv::Mat i2d_mat(){
            return cv::Mat(2, 3, CV_32F, i2d);
        }
    };


    Eigen::MatrixXf change(const Eigen::MatrixXf &r) {
        return r.cwiseMax(r.cwiseInverse());
    }

    Eigen::MatrixXf sz(const Eigen::MatrixXf &w, const Eigen::MatrixXf &h) {
        Eigen::MatrixXf pad = (w + h) * 0.5;
        Eigen::MatrixXf sz2 = (w + pad).cwiseProduct(h + pad);
        return sz2.cwiseSqrt();
    }

    float sz(const float w, const float h) {
        float pad = (w + h) * 0.5;
        float sz2 = (w + pad) * (h + pad);
        return std::sqrt(sz2);
    }

    Eigen::MatrixXf mxexp(Eigen::MatrixXf mx) {
        for (int i = 0; i < mx.rows(); ++i) {
            for (int j = 0; j < mx.cols(); ++j) {
                mx(i, j) = std::exp(mx(i, j));
            }
        }
        return mx;
    }


    class TrackerImpl : public Tracker{
    public:

        ~TrackerImpl() = default;

        bool startup(const std::string &z_path, const std::string &x_path, const std::string &head_path,
                     int gpuid, bool use_multi_preprocess_stream){
            use_multi_preprocess_stream_ = use_multi_preprocess_stream;
            gpu_ = gpuid;
            TRT::set_device(gpuid);

            z_model_ = TRT::load_infer(z_path);
            if(z_model_ == nullptr){
                INFOE("Load model failed: %s", z_path.c_str());
                return false;
            }
            stream_ = z_model_->get_stream();
            z_model_->print();

            x_model_ = TRT::load_infer(x_path);
            if(x_model_ == nullptr){
                INFOE("Load model failed: %s", x_path.c_str());
                return false;
            }
//            x_model_->set_stream(stream_);
            x_model_->print();

            head_model_ = TRT::load_infer(head_path);
            if(head_model_ == nullptr){
                INFOE("Load model failed: %s", head_path.c_str());
                return false;
            }
//            head_model_->set_stream(stream_);
            head_model_->print();

            return true;
        }

        void init(cv::Mat &z_img, cv::Rect &init_bbox) override{
            target_bbox_ = init_bbox;
            cv::Scalar mean = cv::mean(z_img);
            float wc_z = target_bbox_.width + 0.5 * (target_bbox_.width + target_bbox_.height);
            float hc_z = target_bbox_.height + 0.5 * (target_bbox_.width + target_bbox_.height);
            float s_z = std::round(std::sqrt(wc_z * hc_z));  // (s_z)^2=(w+2p)x(h+2p), 模板图像上 不缩放时的 框加上pad 的大小
            cv::Mat z_crop = get_subwindow_tracking(z_img, template_size_, s_z, mean);

            // 绑定输入输出
            zin_ = z_model_->input();
            zf_ = z_model_->output();
            zin_->resize(1, 3, template_size_, template_size_);
            float m[] = {0.406, 0.456, 0.485};
            float std[]  = {0.225, 0.224, 0.229};
            zin_->set_norm_mat_invert(0, z_crop, m, std);
            // 推理
            z_model_->forward(true);

            // 产生hanning窗，以及搜索图像上的grids
            gen_window();
            gen_grids();
        }

        cv::Rect track(cv::Mat& x_img) override{
            cv::Scalar mean = cv::mean(x_img);
            float wc_z = target_bbox_.width + 0.5 * (target_bbox_.width + target_bbox_.height);
            float hc_z = target_bbox_.height + 0.5 * (target_bbox_.width + target_bbox_.height);
            float s_z = std::round(std::sqrt(wc_z * hc_z));  // (s_z)^2=(w+2p)x(h+2p), 模板图像上 不缩放时的 框加上pad 的大小
            float scale_z = float(template_size_) / float(s_z);
            float d_search = float(search_size_ - template_size_) / 2.f;
            float pad = d_search / scale_z;
            float s_x = s_z + 2 * pad;
            cv::Mat x_crop = get_subwindow_tracking(x_img, search_size_, s_x, mean);

            update(x_crop, cv::Size(target_bbox_.width * scale_z, target_bbox_.height * scale_z), scale_z);

            target_bbox_.x = std::max(0, std::min(x_img.cols, target_bbox_.x));
            target_bbox_.y = std::max(0, std::min(x_img.rows, target_bbox_.y));
            target_bbox_.width = std::max(5, std::min(x_img.cols, target_bbox_.width));
            target_bbox_.height = std::max(5, std::min(x_img.rows, target_bbox_.height));

            return target_bbox_;
        }

    private:
        int template_size_ = 128; // 模板图像块大小
        int search_size_ = 256; // 搜索图像块大小
        int total_stride_ = 16; // 最后特征图的总stride
        float penalty_k_ = 0.062; // 惩罚系数
        float window_influence_ = 0.15; // hanning窗权重系数
        float lr_ = 0.765; // 目标框更新的学习率
        int score_size_ = 16; // 最后特征图的大小
        cv::Rect target_bbox_; // 目标框

        shared_ptr<TRT::Infer> z_model_;
        shared_ptr<TRT::Infer> x_model_;
        shared_ptr<TRT::Infer> head_model_;
        shared_ptr<TRT::Tensor> zin_;
        shared_ptr<TRT::Tensor> xin_;
        shared_ptr<TRT::Tensor> zf_;
        shared_ptr<TRT::Tensor> xf_;
        shared_ptr<TRT::Tensor> cls_score_;
        shared_ptr<TRT::Tensor> bbox_pred_;
        TRT::CUStream stream_ = nullptr;
        int gpu_ = 0;
        bool use_multi_preprocess_stream_ = false;

        Eigen::Matrix<float, 16, 16> hanning_win_;
        Eigen::Matrix<float, 16, 16> grid_to_search_x_;
        Eigen::Matrix<float, 16, 16> grid_to_search_y_;
        Eigen::Matrix<float, 16, 16> pred_score_;
        Eigen::Matrix<float, 16, 16> pred_x1_;
        Eigen::Matrix<float, 16, 16> pred_x2_;
        Eigen::Matrix<float, 16, 16> pred_y1_;
        Eigen::Matrix<float, 16, 16> pred_y2_;
        Eigen::Matrix<float, 16, 16> s_c_;
        Eigen::Matrix<float, 16, 16> r_c_;
        Eigen::Matrix<float, 16, 16> penalty_;
        Eigen::Matrix<float, 16, 16> pscore_;


        void update(cv::Mat &x_crop, cv::Size_<float> target_sz, float scale_z){
            // 绑定输入输出
            xin_ = x_model_->input();
            xf_ = x_model_->output();
            xin_->resize(1, 3, search_size_, search_size_);
            float m[] = {0.406, 0.456, 0.485};
            float std[]  = {0.225, 0.224, 0.229};
            xin_->set_norm_mat_invert(0, x_crop, m, std);
            // 推理
            x_model_->forward(true);

            // 绑定输入输出
            head_model_->input(0) = zf_;
            head_model_->input(1) = xf_;
            cls_score_ = head_model_->output(1);
            bbox_pred_ = head_model_->output(0);
            // 推理
            head_model_->forward(true);

            auto *cls_ptr = cls_score_->cpu<float>();
            auto *bbox_ptr = bbox_pred_->cpu<float>();

            for (int i = 0; i < 5; ++i) {
                for (int j = 0; j < 16; ++j) {
                    for (int k = 0; k < 16; ++k) {
                        if (i == 0)
                            pred_x1_(j, k) = bbox_ptr[i * 256 + j * 16 + k];
                        else if (i == 1)
                            pred_y1_(j, k) = bbox_ptr[i * 256 + j * 16 + k];
                        else if (i == 2)
                            pred_x2_(j, k) = bbox_ptr[i * 256 + j * 16 + k];
                        else if (i == 3)
                            pred_y2_(j, k) = bbox_ptr[i * 256 + j * 16 + k];
                        else
                            pred_score_(j, k) = cls_ptr[j * 16 + k];
                    }
                }
            }

            pred_x1_ = grid_to_search_x_ - pred_x1_;
            pred_y1_ = grid_to_search_y_ - pred_y1_;
            pred_x2_ = grid_to_search_x_ + pred_x2_;
            pred_y2_ = grid_to_search_y_ + pred_y2_;

            // size penalty
            s_c_ = change(
                    sz(pred_x2_ - pred_x1_, pred_y2_ - pred_y1_) / sz(target_sz.width, target_sz.height));  // scale penalty
            r_c_ = change((target_sz.width / target_sz.height) /
                          ((pred_x2_ - pred_x1_).array() / (pred_y2_ - pred_y1_).array()).array());  // ratio penalty

            penalty_ = mxexp(-(r_c_.cwiseProduct(s_c_) - Eigen::MatrixXf::Ones(16, 16)) * penalty_k_);
            pscore_ = penalty_.cwiseProduct(pred_score_);

            // window penalty
            pscore_ = pscore_ * (1 - window_influence_) + hanning_win_ * window_influence_;

            // get max
            Eigen::MatrixXd::Index maxRow, maxCol;
            pscore_.maxCoeff(&maxRow, &maxCol);

            // to real size
            float x1 = pred_x1_(maxRow, maxCol);
            float y1 = pred_y1_(maxRow, maxCol);
            float x2 = pred_x2_(maxRow, maxCol);
            float y2 = pred_y2_(maxRow, maxCol);

            float pred_xs = (x1 + x2) / 2.f;
            float pred_ys = (y1 + y2) / 2.f;
            float pred_w = x2 - x1;
            float pred_h = y2 - y1;

            float diff_xs = pred_xs - float(search_size_) / 2.f;
            float diff_ys = pred_ys - float(search_size_) / 2.f;

            diff_xs = diff_xs / scale_z;
            diff_ys = diff_ys / scale_z;
            pred_w = pred_w / scale_z;
            pred_h = pred_h / scale_z;

            target_sz.width = std::round(float(target_sz.width) / scale_z);
            target_sz.height = std::round(float(target_sz.height) / scale_z);

            // size learning rate
            float lr = penalty_(maxRow, maxCol) * pred_score_(maxRow, maxCol) * lr_;

            // size rate
            float res_xs = target_bbox_.x + target_bbox_.width / 2.f + diff_xs;
            float res_ys = target_bbox_.y + target_bbox_.height / 2.f + diff_ys;
            float res_w = pred_w * lr + (1 - lr) * target_sz.width;
            float res_h = pred_h * lr + (1 - lr) * target_sz.height;

            target_bbox_.width = std::round(target_sz.width * (1 - lr) + lr * res_w);
            target_bbox_.height = std::round(target_sz.height * (1 - lr) + lr * res_h);
            target_bbox_.x = std::round(res_xs - target_bbox_.width / 2.f);
            target_bbox_.y = std::round(res_ys - target_bbox_.height / 2.f);

            cout << "**************** DEBUG ****************" << endl;
            cout << "bbox x " << target_bbox_.x << endl;
            cout << "bbox y " << target_bbox_.y << endl;
            cout << "bbox w " << target_bbox_.width << endl;
            cout << "bbox h " << target_bbox_.height << endl;
        }


        cv::Mat get_subwindow_tracking(cv::Mat &img, int model_sz, float original_sz, const cv::Scalar &avg_chans) {
            cv::Mat img_patch_ori;  // 填充不缩放
            cv::Mat img_patch;  // 填充+缩放之后输出的最终图像块
            cv::Rect crop;
            float c = (original_sz + 1) / 2.f;
            // 计算出剪裁边框的左上角和右下角
            int context_xmin = std::round(target_bbox_.x + target_bbox_.width / 2.f - c);
            int context_xmax = std::round(context_xmin + original_sz - 1);
            int context_ymin = std::round(target_bbox_.y + target_bbox_.height / 2.f - c);
            int context_ymax = std::round(context_ymin + original_sz - 1);
            // 边界部分要填充的像素
            int left_pad = std::max(0, -context_xmin);
            int top_pad = std::max(0, -context_ymin);
            int right_pad = std::max(0, context_xmax - img.cols + 1);
            int bottom_pad = std::max(0, context_ymax - img.rows + 1);
            // 填充之后的坐标
            crop.x = context_xmin + left_pad;
            crop.y = context_ymin + top_pad;
            crop.width = context_xmax + left_pad - crop.x;
            crop.height = context_ymax + top_pad - crop.y;
            // 填充像素
            if (left_pad > 0 || top_pad > 0 || right_pad > 0 || bottom_pad > 0) {
                cv::Mat pad_img = cv::Mat(img.rows + top_pad + bottom_pad, img.cols + left_pad + right_pad, CV_8UC3, avg_chans);
                for (int i = 0; i < img.rows; ++i) {
                    memcpy(pad_img.data + (i + top_pad) * pad_img.cols * 3 + left_pad * 3, img.data + i * img.cols * 3,
                           img.cols * 3);
                }
                pad_img(crop).copyTo(img_patch_ori);
            } else
                img(crop).copyTo(img_patch_ori);
            if (img_patch_ori.rows != model_sz || img_patch_ori.cols != model_sz)
                cv::resize(img_patch_ori, img_patch, cv::Size(model_sz, model_sz));
            else
                img_patch = img_patch_ori;

            return img_patch;
        }


        void gen_window() {
            cv::Mat window(16, 16, CV_32FC1);
            cv::createHanningWindow(window, cv::Size(16, 16), CV_32F);
            cv::cv2eigen(window, hanning_win_);
        }

        void gen_grids() {
            Eigen::Matrix<float, 1, 16> grid_x;
            Eigen::Matrix<float, 16, 1> grid_y;

            for (int i = 0; i < 16; ++i) {
                grid_x[i] = float(i) * 16;
                grid_y[i] = float(i) * 16;
            }
            grid_to_search_x_
                << grid_x, grid_x, grid_x, grid_x, grid_x, grid_x, grid_x, grid_x, grid_x, grid_x, grid_x, grid_x, grid_x, grid_x, grid_x, grid_x;
            grid_to_search_y_
                << grid_y, grid_y, grid_y, grid_y, grid_y, grid_y, grid_y, grid_y, grid_y, grid_y, grid_y, grid_y, grid_y, grid_y, grid_y, grid_y;
        }

    };


    shared_ptr<Tracker> create_tracker(
            const std::string &z_path, const std::string &x_path, const std::string &head_path,
            int gpuid, bool use_multi_preprocess_stream
    ){
        shared_ptr<TrackerImpl> instance(new TrackerImpl{});
        if(!instance->startup(z_path, x_path, head_path, gpuid, use_multi_preprocess_stream))
            instance.reset();
        return instance;
    }

}
