#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <map>
#include <set>
#include <unordered_set>
#include <unistd.h>
#include "framework.hpp"
#include "utils/json_helpers.hpp"

using namespace std;
using namespace neuro;
using nlohmann::json;

typedef runtime_error SRE;

typedef vector <double>  Vec1D;
typedef vector < vector <double> >  Vec2D;
typedef vector < vector < vector <double> > >  Vec3D;
typedef vector < vector < vector < vector <double> > > > Vec4D;

typedef vector <int> VecInt1D;
typedef vector < vector <int> > VecInt2D;
typedef vector < vector < vector <int> > >  VecInt3D;
typedef vector < vector < vector < vector <int> > > > VecInt4D;

enum LayerType { 
  DenseLayer,
  Conv2DLayer,
  MaxPooling2DLayer,
  FlattenLayer,
  SoftmaxDecoderLayer,
  InputLayer,
};

class Layer;
class Dense;
class Flatten;
class Conv2D;
class MaxPooling2D;
class SoftmaxDecoder;
class Input;

class Model {

public:
  Model(); // TODO done
  ~Model(); /**< Free memory */ // TODO done

  /** Create the model by setting up the layers vector. */            
  void create_model(const json &p_weights,
                    const json &input_shape_j,
                    const vector <string> &layer_type_vec,
                    const Vec1D &thresholds_vec,
                    const double &div,
                    const bool &is_preprocessing);

  void calculate();

  void write_network(const string &path);

  Vec3D data_inputs;   /**< Input values from cml */

protected:

  /** It basically reads a weight array and layer type array, create the layer class with corresponding layer type, 
      and put them into layers vector.
   *  @param p_weights JSON array. 
   */
  void load_weights(const json &p_weights); // TODO done

  Layer *add_layer(Dense *dense);                 /**< Add a dense layer */
  // Layer *add_layer(Conv2D *conv2d);               /**< Add a Conv2D layer */
  // Layer *add_layer(Flatten *flatten);             /**< Add a flatten layer */
  // Layer *add_layer(MaxPooling2D *pooling);        /**< Add a Max Pooling layer */
  // Layer *add_layer(SoftmaxDecoder *softmax);      /**< Add a softmax decoder layer */
  Layer *add_layer(Input *input);                 /**< Add an input layer */

  vector <class Layer*> layers;  /**< Store all instance of layers */
  vector <string> layer_types;   /**< Layer types */
  VecInt2D extra_layer_connections; /**< Extra connections for residual block */
  Vec1D zero_out;                /**< The fraction of weights we zero out for each layer */
  Vec1D thresholds;              /**< Thresholds for input/dense/conv2d layer if using binary activation function */
 
  VecInt3D A;                    /**< Input neuron ids */

  VecInt1D input_shape;          /**< Input shape */

  Network *net;                  /**< The neuromorphic network */
  Node *head_bn;                 /**< The head of the bias neuron */

  string proc_name;              /**< Processor name */
  json proc_params;              /**< Processor params */
  string howto;                  /**< One of regular|binary|neuromorphic|reshape */
  string model_summary;          /**< The summary of the model */

  json params;                   /**< A copy of json params from from_json() */

  map <int, string> node_names;  /**< Node id to node name. */
  
  double overall_zero_out;       /**< The fraction of weights we zero out considering all layers */
  int total_sim_time;            /**< The simulation time for neuromorphic processor */
  int integration_time;          /**< The integration time for neuromorphic processor */
  size_t total_params;           /**< The total number of params that are involved  */
  int regular_starting_layer;    /**< The starting layer index of performing regular operation */
   
  bool show_spike_info;          /**< True when showing output neurons' spiking time */
  bool preprocessing;            /**< True when preprocessing the input by feeding forward one layer */
  bool print;                    /**< True when printing outputs */
  bool suppress_hidden_outputs;  /**< True when suppress_hidden_outputs */

};


/** Layer class */
class Layer {
  public:
    
    /** Initialize the member variables */
    Layer(const LayerType type); // TODO done  

    /** Free the memory */ // TODO done
    ~Layer();

    /** Set the divisor for conv2d and dense layer if --divisor option is used in the command-line 
      * @param divisor the value of command-line argument --divisor.
    */
    void set_divisor(double divisor);       

    /** Make a traslated call based on the layer type. It also collectes layer info */
    // Vec3D get_neuromorphic_outputs(const Network *net, Processor *p);

    /** Make a traslated call based on the layer type. It also collectes layer info */
    Vec3D calculate_regular(const Vec3D &inputs, bool binary_output, bool collect_layer_info = true);

    /** Make a traslated call based on the layer type. It reurns neurons id */ 
    // VecInt3D calculate_neuromorphic(const VecInt3D &A, 
    //                                 Network *net, 
    //                                 map <int, string> &node_names, 
    //                                 Node* bn, 
    //                                 int &total_sim_time,
    //                                 int skip_layer_delay);

    VecInt3D calculate_neuromorphic(const VecInt3D &A, 
                                    Network *net, 
                                    map <int, string> &node_names, 
                                    Node* bn, 
                                    int &total_sim_time,
                                    int skip_layer_delay);

    /** Given the inpput shape, we set the output shape */
    void set_output_shape(const VecInt1D &input_shape);

    LayerType layer_type;         /**< The layer type */
    string layer_info;            /**< Layer information (weight shape, output shape, neurons #, etc. ) */
    string name;                  /**< The name of layer in the format of layerType_index(for example dense_2 ) */
    string dot_attr;              /**< the output for graphviz dot tool */
    double threshold;             /**< The threshold of neruons */
    double zero_out;              /**< The zero out fraction of weights and biases */
    VecInt1D input_shape;         /**< Iutput shape */
    VecInt1D output_shape;        /**< Output shape */
    VecInt1D weight_shape;        /**< Weight shape */
    int sim_time;                 /**< Simulation time */


    /** Each Layer stores one specific layer type */
    union {
      Dense *dense;
      // Conv2D *conv2d;
      // Flatten *flatten;
      // MaxPooling2D *max_pooling2d;
      // SoftmaxDecoder *softmax_decoder;
      Input *input;
    };

};

/** This is more like a dummy class. It's only for collecting input layer info easily. */
class Input {
  public:
    Input(bool p_preprocessing); // TODO done
    bool preprocessing;
};

/** Dense layer */
class Dense {
  public:

    /** 
     * Initialize the dense layer 
     * @param p_weights 2D double vector.
     * @param p_biases 1D double vector. If the size is 0, the biases of each output neurons is 0.
     * @param p_thresholds 1D double vector. If the size is 0, the thresholds of each output neurons is 0.
     * @param p_output_layer If true, this layer is the last layer of neuromorphic network.
     */
    Dense(const Vec2D &p_weights, const Vec1D &p_biases, const Vec1D &p_thresholds, bool p_output_layer); // TODO done

    Vec1D calculate_regular(const Vec1D &inputs, bool binary_output);

    VecInt1D calculate_neuromorphic(const VecInt1D &A, 
                                    Network *net, 
                                    map <int, string> &node_names, 
                                    Node* bn, 
                                    int &sim_time, 
                                    int skip_layer_delay);

     /** 
     * Set the divisor rather than use default divisor
     * @param divisor divisor for thresholds, weights and biases. 
     */
    void set_divisor(double divisor);

    /** 
     * Return the ouput shape given the input shape
     * @param input_shape %Input shape
     * @return Calculate output shape based on the input_shape and weight shape, and then return it.
     */

    VecInt1D get_output_shape(const VecInt1D &input_shape);

   
  
    size_t num_params;   /**< Number of params belong to this dense layer */
    size_t num_neurons;  /**< Number of neurons belong to this dense layer */
    size_t num_synapses; /**< Number of synapses belong to this dense layer */
    int layer_index;     /**< The index of this dense layer in model->layers */

  protected:

    /** 
     * We read min/max weights and thresholds from network's node and edges's property.
     * Then, we find the largest weight_divisor and threshold_divisor separately. 
     * The final divisor is the max of weight_divisor and threshold_divisor.
     * @param net Neuromorphic network
     * @return The the largest possible divisor value 
     */
    double get_divisor(const Network *net);

    void scale_weights_and_thresholds(const Network *net);

    VecInt1D B;        /**< output neuron ids */
    
    bool output_layer; /**< If true, this is a output layer of neuromorphic network */

    Vec1D thresholds;  /**< Threshold values */
    Vec1D biases;      /**< Bias values */
    Vec2D weights;     /**< weight values */

    double divisor;    /**< Divisor for weights/biases/thresholds */
};
