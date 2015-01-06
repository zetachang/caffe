#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void TripletSimilarityPrecisionLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The query and similar data should have the same number.";
  CHECK_EQ(bottom[0]->num(), bottom[2]->num())
      << "The query and dissimilar data should have the same number.";
  (*top)[0]->Reshape(1, 1, 1, 1);
}

template <typename Dtype>
void TripletSimilarityPrecisionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
        
  Dtype accuracy = 0;
  
  Blob<Dtype> diffSimilar = new Blob<Dtype>;
  Blob<Dtype> diffDissimilar = new Blob<Dtype>;
  
  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;
  
  caffe_sub(count, bottom[0]->cpu_data(), bottom[1]->cpu_data(), diffSimilar.mutable_cpu_data());
  caffe_sub(count, bottom[0]->cpu_data(), bottom[2]->cpu_data(), diffDissimilar.mutable_cpu_data());
  
  for (int i = 0; i < num; i++) {
    Dtype dist1 = caffe_cpu_dot(dim, diffSimilar[i * dim], diffSimilar[i * dim]);
    Dtype dist2 = caffe_cpu_dot(dim, diffDissimilar[i * dim], diffDissimilar[i * dim]);
    if (dist1 < dist2) {
      ++accuracy;
    }
  }
  // LOG(INFO) << "Similarity precision: " << accuracy;
  (*top)[0]->mutable_cpu_data()[0] = accuracy / num;
  // Accuracy layer should not be used as a loss function.
  
  delete diffSimilar;
  delete diffDissimilar;
}

INSTANTIATE_CLASS(TripletSimilarityPrecisionLayer);

}  // namespace caffe