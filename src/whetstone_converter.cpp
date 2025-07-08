#include "whetstone.hpp"

using namespace std;

#ifndef debug_print
#define debug_print false
#endif

// Commands:
// weights (path to weights) DONE
// layers (ignore "spiking" layers (whetstone.layers.spiking_brelu and whetstone.layers.spiking_sigmoid) DONE
// thresholds (something to do with spiking brelu) DONE
// input_shape DONE
// network (write out) EASY
// divisor (divisor lmao) ? see chao hui's implementation of decoding. MAYBE
// summary (will be done by default)
// help (me please)
// make (this can only be ran when layers, thresholds, weights, input_shape are set.

void print_commands(FILE *f) 
{ 

  fprintf(f, "This is a Whetstone network converter program. The commands listed below are case-insensitive,\n");
  fprintf(f, "For commands that take a json either put a filename on the same line,\n");
  fprintf(f, "or the json can be multiple lines, starting on the next line.\n\n");


  fprintf(f, "Action commands --\n");

  fprintf(f, "WEIGHTS path/to/whetstone/weights/json          - Path to Whetstone weights file\n");
  fprintf(f, "INPUT_SHAPE/IS input_shape_json                 - Json array of integers representing the input shape\n");
  fprintf(f, "LAYER_TYPES/LT layer_types_json                 - Json array of layer types. Supported types: dense, conv2d-same, conv2d-valid, maxpooling-n, flatten, softmax_decoder\n");
  fprintf(f, "THRESHOLDS/T thresholds_json                    - Json array of floats representing thresholds for each layer. If there are less thresholds than layers, the final listed threshold will be applied to the remaining layers\n");
  fprintf(f, "DIVISOR/D                                       - The divisor to weights & thresholds"); 
  fprintf(f, "PREPROCESSING/P                                 - Preprocess the input by doing one layer feed-forward"); 
  fprintf(f, "BUILD                                           - Create the spiking neural network. Must set weights, input_shapes, and layer_types first\n");
  fprintf(f, "SAVE path/to/network/json                       - Saves built spiking neural network to the specified path\n");
  fprintf(f, "\n");

  fprintf(f, "Info commands --\n");
  fprintf(f, "?                                   - Print commands\n");
  fprintf(f, "Q                                   - Quit\n");
}

/**
  Return the depth of the json array
*/
static int check_json_array_depth(const json &j) {

  int depth = 0;
  const json *tmp;

  tmp = &j;
  while (tmp->is_array() && tmp->size() != 0) {
    depth++;
    tmp = &(tmp->at(0));
  }

  return depth;
}

/**
  Convert json to 1D vector
*/
static Vec1D json_to_vec1d(const json &arr) {
  if (check_json_array_depth(arr) != 1) throw SRE("json_to_vec1d: json depth != 1");
  return arr.get<Vec1D>();
}

/**
  Convert json to 2D vector
*/
static Vec2D json_to_vec2d(const json &arr) {
  Vec2D vec;
  size_t i, cols;

  if (check_json_array_depth(arr) != 2) throw SRE("json_to_vec2d: json depth != 2");

  cols = arr[0].size();
  for (i = 0; i < arr.size(); i++) {
    if (arr[i].size() != cols)  throw SRE("json_to_vec2d: sub-arrays have different sizes.");
    vec.push_back(json_to_vec1d(arr[i]));
  }

  return vec;
}

/**
  Convert json to 3D vector
*/
static Vec3D json_to_vec3d(const json &arr) {
  Vec3D vec;
  size_t i, size;

  if (check_json_array_depth(arr) != 3) throw SRE("json_to_vec3d: json depth != 3");

  size = arr[0].size();
  for (i = 0; i < arr.size(); i++) {
    if (arr[i].size() != size) throw SRE("json_to_vec3d: sub-arrays have different sizes");
    vec.push_back(json_to_vec2d(arr[i]));
  }

  return vec;
}

/**
  Convert json to 4D vector
*/
static Vec4D json_to_vec4d(const json &arr) {
  Vec4D vec;
  size_t i, size;

  if (check_json_array_depth(arr) != 4) throw SRE("json_to_vec3d: json depth != 4");

  size = arr[0].size();
  for (i = 0; i < arr.size(); i++) {
    if (arr[i].size() != size) throw SRE("json_to_vec4d: sub-arrays have different sizes");
    vec.push_back(json_to_vec3d(arr[i]));
  }

  return vec;
}

static Vec1D zeros(int num) {
  Vec1D vec;
  if (num <= 0) throw SRE("ones: num must > 0");
  vec.resize(num, 0);
  return vec;
}

static void resize_vec_int_3d(VecInt3D &v, int rows, int cols, int depth, int value) {
  int i, j;

  if (rows <= 0) throw SRE("resize_vec3d: rows <= 0");
  v.clear();
  v.resize(rows);
  if (cols > 0) {
    for (i = 0; i < rows; i++) v[i].resize(cols);
    if (depth > 0) {
      for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) v[i][j].resize(depth, value);
      }
    }
  }
}

static void resize_vec3d(Vec3D &v, int rows, int cols, int depth, double value) {
  int i, j;

  if (rows <= 0) throw SRE("resize_vec3d: rows <= 0");
  v.clear();
  v.resize(rows);

  if (cols > 0) {
    for (i = 0; i < rows; i++) v[i].resize(cols);
    if (depth > 0) {
      for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) v[i][j].resize(depth, value);
      }
    }
  }
}

static double clamp_value(double w_or_t, int min_value, int max_value, bool discrete = true) {
  double rv = (discrete) ? rint(w_or_t) : w_or_t;

  if (rv < min_value) return min_value;
  if (rv > max_value) return max_value;

  return rv;
}

static void add_and_set_edge(Network *net, int from, int to, int delay, double weight) {
  Edge *e = net->add_edge(from, to);
  e->set("Delay", delay);
  e->set("Weight", weight);
}

static void split_edge_delay(Network *net,
                             int from,
                             int to,
                             int delay,
                             double weight,
                             map <int, string> &node_names) {
  int max_delay;
  int i, j, num_split;
  bool exist;
  bool output_neuron;
  string base_name, name;
  const Property *pr;
  Node *n, *from_n;
  char buf[110];

  // output_neuron = net->get_data("other")["make_int_outputs"];
  output_neuron = false;
  pr = net->get_edge_property("Delay");
  max_delay = pr->max_value;

  num_split = delay / max_delay;
  if (delay % max_delay == 0) num_split--;

  base_name = node_names[from];
  for (i = 0; i < num_split; i++) {
    snprintf(buf, 110, "-%d", i+1);
    name = base_name + buf;
    from_n = net->get_node(from);
    exist = false;

    /* Check if the intermediate node exists.
       This should be quick. If it has a intermediate neuron,
       the size of outgoing edges should be one except for the last intermediate node.
    */
    for (j = 0; j < (int) from_n->outgoing.size(); j++) {

      n = from_n->outgoing[j]->to;
      if (node_names[n->id] == name) {
        exist = true;
        break;
      }
    }
    if (!exist) {
      n = net->add_node(net->num_nodes());
      node_names[n->id] = name;
      if (output_neuron) net->add_output(n->id);
      add_and_set_edge(net, from, n->id, max_delay, 1);
    }

    from = n->id;
  }

  add_and_set_edge(net, from, to, delay - max_delay * num_split, weight);
}

static void shape_to_cstr(char *buf, int bufsize, const VecInt1D &shape) {
  switch(shape.size()) {
    case 0:
      snprintf(buf, bufsize, "N/A"); break;
    case 1:
      snprintf(buf, bufsize, "(%d)", shape[0]); break;
    case 2:
      snprintf(buf, bufsize, "(%d, %d)", shape[0], shape[1]); break;
    case 3:
      snprintf(buf, bufsize, "(%d, %d, %d)", shape[0], shape[1], shape[2]); break;
    case 4:
      snprintf(buf, bufsize, "(%d, %d, %d, %d)", shape[0], shape[1], shape[2], shape[3]); break;
    default:
      throw SRE("shape_to_cstr(): shape's # elements must be <= 4");
  }
}

json weights_to_json(const string& path) 
{
  json j;
  ifstream weights_file;
 
  weights_file.open(path);

  if (!weights_file) {
      throw SRE("Failed to open " + path);
  }

  weights_file >> j;

  return j;
}

bool read_json(const vector <string> &sv, size_t starting_field, json &rv)
{
  bool success;
  string s;
  ifstream fin;

  rv.clear();
  if (starting_field < sv.size()) {
  fin.clear();
  fin.open(sv[starting_field].c_str());
  if (fin.fail()) {
    perror(sv[starting_field].c_str());
    return false;
  }
  try { fin >> rv; success = true; } catch(...) { success = false; }
  fin.close();
  return success;

  } else {
  try {
    cin >> rv;
    getline(cin, s);
    return true;
  } catch (...) {
    return false;
  }
  }
}

bool is_number(const string& str) {
  size_t i;

  for (i = 0; i < str.size(); i++) {
    if (std::isdigit(str[i]) == 0) return false;
  }

  return true;
}

void to_uppercase(string &s) 
{
  size_t i;
  for (i = 0; i < s.size(); i++) {
    if (s[i] >= 'a' && s[i] <= 'z') {
      s[i] = s[i] + 'A' -'a';
    }
  }
}

int main(int argc, char **argv) 
{

  Model *m;

  istringstream ss;
  ofstream fout;
  vector <string> sv, keys;
  vector <double> dv;
  string s, l, weights_string;
  string prompt, cmd;

  vector <double> thresholds;
  vector <string> layer_types;
  json weights, tmp_j, input_shape;
  double div, dval;
  bool w_assigned, i_assigned, l_assigned, is_preprocessing, m_built;

  MOA rng;
  uint32_t seed;

  size_t i, size;

  if (argc > 2 || (argc == 2 && strcmp(argv[1], "--help") == 0)) {
    fprintf(stderr, "usage: whetstone_converter [prompt]\n");
    fprintf(stderr, "\n");
    print_commands(stderr);
    exit(1);
  }

  seed = rng.Seed_From_Time();
  rng.Seed(seed, "whetstone_converter");

  if (argc == 2) {
    prompt = argv[1];
    prompt += " ";
  }

  m = new Model;

  w_assigned = i_assigned = l_assigned = is_preprocessing = m_built = false;

  thresholds = {0.5};
  div = -1;

  while(1) {
    try {
      if (prompt != "") printf("%s", prompt.c_str());
      if (!getline(cin, l)) return 0;
      sv.clear();
      ss.clear();
      ss.str(l);
      while (ss >> s) sv.push_back(s);

      size = sv.size();
      if (size != 0) to_uppercase(sv[0]);

      if (size == 0) {
      } else if (sv[0][0] == '#') {
      } else if (sv[0] == "?") {
          print_commands(stdout);
      } else if (sv[0] == "Q") {
          exit(0);
      } else if (sv[0] == "WEIGHTS" || sv[0] == "W") {
          if(!read_json(sv, 1, weights)) printf("usage: WEIGHTS/W weights_json; Bad json\n");
          else w_assigned = true;
      } else if (sv[0] == "INPUT_SHAPE" || sv[0] == "IS") {
          if(!read_json(sv, 1, weights)) printf("usage: INPUT_SHAPE/IS input_shape_json: Bad json\n");
          else i_assigned = true;
      } else if (sv[0] == "LAYER_TYPES" || sv[0] == "LT") {
          if(!read_json(sv, 1, tmp_j)) printf("usage: LAYER_TYPES/LT layer_types_json: Bad json\n");
          else {
            layer_types.clear();
            for (i = 0; i < tmp_j.size(); i++) {
                layer_types.push_back(tmp_j[i]);
            }

            tmp_j.clear();

            l_assigned = true;
        }

      } else if (sv[0] == "THRESHOLDS" || sv[0] == "T") {
          if(!read_json(sv, 1, tmp_j)) printf("usage: THRESHOLDS/T thresholds_json_json: Bad json\n");
          else {
            for (i = 0; i < tmp_j.size(); i++) {
              try {
                dval = tmp_j[i].get<double>();
              } catch (...) {
                throw SRE("THRESHOLDS/T thresholds_json: Bad value"); 
              }
            }

            thresholds.clear();

            for (i = 0; i < tmp_j.size(); i++) thresholds.push_back(tmp_j[i].get<double>());
            tmp_j.clear();
          }

      } else if (sv[0] == "DIVISOR" || sv[0] == "D") {

         if (sscanf(sv[1].c_str(), "%lf", &div) != 1) {
             printf("Bad divisor: %s\n", sv[1].c_str());
         }

      } else if (sv[0] == "PREPROCESSING" || sv[0] == "P") {

          is_preprocessing = !is_preprocessing;

          printf("Preprocessing is now %s\n", is_preprocessing ? "true" : "false");

      } else if (sv[0] == "BUILD") {

          if (!w_assigned && !i_assigned && !l_assigned) printf("Must set weights, input_shape, and layer_types before calling build\n");
          else if (!i_assigned && !l_assigned) printf("Must set input_shape, and layer_types before calling build\n");
          else if (!w_assigned && !l_assigned) printf("Must set weights, and layer_types before calling build\n");
          else if (!w_assigned && !i_assigned) printf("Must set weights, and input_shape before calling build\n");
          else if (!l_assigned) printf("Must set layer_types before calling build\n");
          else if (!i_assigned) printf("Must set input_shape before calling build\n");
          else if (!w_assigned) printf("Must set weights before calling build\n");
          else {
              m->create_model(weights, input_shape, layer_types, thresholds, div, is_preprocessing);
              m->calculate();
              m_built = true;
          }
      } else if (sv[0] == "SAVE") {
          if (!m_built) {
              printf("Must build the network before saving\n");
          } else {
              m->write_network(sv[1]);
          }
      } else {
          printf("Invalid command %s.  Use '?' to print a list of commands.\n", sv[0].c_str());
      }
    } catch (const std::exception &e) {
      printf("%s\n", e.what());
    }
  }
}


Model::Model(){
  net = nullptr;
};

Model::~Model() {
  size_t i;
  for (i = 0; i < layers.size(); i++) delete layers[i];
  if (net != nullptr) delete net;
}

void Model::create_model(const json &p_weights,
                         const json &input_shape_j,
                         const vector <string> &layer_type_vec,
                         const Vec1D &thresholds_vec,
                         const double &div,
                         const bool &is_preprocessing) {
  size_t size;
  size_t i, j, d;
  int num_neurons, depth;
  char buf[200];
  VecInt1D shape;
  json tmp_j;

  preprocessing = is_preprocessing;
  layer_types = std::move(layer_type_vec);
  thresholds = std::move(thresholds_vec);

  tmp_j = input_shape_j;

  // Set data_input (taken from Model::from_json()
  if (check_json_array_depth(tmp_j) != 1) throw SRE("Input shape must be a 1-D non-empty array");
  if (tmp_j.size() > 3) throw SRE("Input shape's dim must be <= 3");

  for (i = 0; i < tmp_j.size(); i++) {
    if (!tmp_j[i].is_number()) throw SRE("Input shape's element is not a int number");
    if (tmp_j[i].get<int>() != tmp_j[i].get<double>()) throw SRE("Input shape's element must be a int number");
        input_shape.push_back(tmp_j[i]);
    }

  if (input_shape.size() == 2) input_shape.push_back(1); // conv inputs

  /* resize the inputs properly */
  if (input_shape.size() == 1) resize_vec3d(data_inputs, 1, 1, input_shape[0], 0);
  else resize_vec3d(data_inputs, input_shape[0], input_shape[1], input_shape[2], 0);

  // Load processor
  Processor *p;
  json proc_params = json_from_string_or_file("params/risp.json");
  p = Processor::make("risp", proc_params);

  net = new Network();
  net->set_properties(p->get_network_properties());
  net->set_data("proc_params", json_from_string_or_file("params/risp.json"));

  load_weights(p_weights);
  if (debug_print) cout << "Finish loading weights" << endl;

  // not always true, must have 2 or more layers for preprocessing
  if (preprocessing) shape = layers[1]->output_shape;
  else shape = layers[0]->output_shape;

  size = shape.size();

  if (layers.size() <= 1) throw SRE("Model - create_model(): no layers other than input layer are creeated. Empty weight file");

  /* create input neurons for the neuromorphic network */
  num_neurons = net->num_nodes();

  if (div != -1) {
    for (i = 0; i < layers.size(); i++) layers[i]->set_divisor(div);
  }

  if (size == 1) {
    resize_vec_int_3d(A, 1, 1, shape[0], 0);
  } else {
    resize_vec_int_3d(A, shape[0], shape[1], shape[2], 0);
  }

  for (i = 0; i < A.size(); i++) {
    for (j = 0; j < A[0].size(); j++) {
        for (d = 0; d < A[0][0].size(); d++) {
          A[i][j][d] = num_neurons;
          net->add_node(num_neurons);

          if (size == 1) snprintf(buf, 200, "A(%zu)", d);
          else snprintf(buf, 200, "A(%zu,%zu,%zu)", i, j, d);

          node_names[num_neurons] = buf;
          net->add_input(num_neurons);
          num_neurons++;
        }
      }
    }

  if (debug_print) cout << "Finish creating model" << endl;
}

void Model::calculate() {
  size_t i, j, starting_layer;

  vector <VecInt3D> all_node_ids;
  Vec4D all_outputs;

  VecInt1D all_sim_time;

  Layer *l;
  char buf[400];
  int summary_len;
  int dummy_sim_time;
  int from_index;

  string add_layer_info;
  string s;

  EdgeMap::iterator eit;
  NodeMap::iterator nit;
  Edge *e;
  Node *n, *from, *to;


  dummy_sim_time = -1;
  total_sim_time = (true) ? 1 : 0; // TODO: Go here
  from_index = -1;
  summary_len = -1;

  if (summary_len == -1) summary_len = 130;

  /* push inputs first */
  all_outputs.push_back(data_inputs);
  all_node_ids.push_back(A);
  all_sim_time.push_back(total_sim_time);

  /* summary header  */
  model_summary = "";
  model_summary += std::string(summary_len, '-') + '\n';
  snprintf(buf, 400, "%-30s%-20s%-20s%-12s%-12s%-12s%-12s%-12s%-12s",
               "Layer (Type)", "Weight Shape", "Output Shape", "Neuron #", "Synapse #", "Param #", "Threshold", "Sim Time", "Zero Out");

  s = (string) buf; s.resize(summary_len);
  model_summary += s + '\n';
  model_summary += std::string(summary_len, '=') + '\n';

  /* input layer summary */
  l = layers[0];

  l->calculate_neuromorphic(A, net, node_names, nullptr, total_sim_time, 0);
  l->layer_info.resize(summary_len);
  model_summary += l->layer_info + '\n';

  if (debug_print) cout << "Finish processing layer " << l->name << endl;

  /* Get the actural input layer for neuromorphic network.
     When preprocessing the input, the first hidden layer is the actual input layer. */
  starting_layer = 1;
  // if (preprocessing && howto == "neuromorphic") {
  // false right now
  if (preprocessing) {
    l = layers[1];
    (void) l->calculate_regular(data_inputs, true);
    snprintf(buf, 400, "%-12d%-12d", (int) net->num_nodes(), (int) net->num_edges());
    l->layer_info.replace(70, 24, buf);
    l->layer_info.resize(summary_len);
    model_summary += std::string(summary_len, '-') + '\n';
    model_summary += l->layer_info + '\n';

    all_node_ids.push_back(A);
    all_sim_time.push_back(total_sim_time);
    starting_layer = 2;

    if (debug_print) cout << "Finish processing layer " << l->name << endl;
  }

  /* Do the calculation. Each layer does its own calculation independently based on the inputs
     and return the outputs.  Conv/Pooling/Flatten layers take 3D input. Dense layer takes 1D input.
     When reading in layer types, we have done error checking to make sure the previous
     layer's output match to next layer's input.
  */
  from = nullptr;
  to = nullptr;
  for (i = starting_layer; i < layers.size(); i++) {
    add_layer_info = "";
    l = layers[i];

    /* Create and connect the bias neurons for dense and conv2d layers.
         The delay is the sim_time for that layer
    */
    if (l->layer_type == LayerType::DenseLayer || l->layer_type == LayerType::Conv2DLayer) {

    to = net->add_node(net->num_nodes());
    // if (params["make_bn_outputs"]) net->add_output(to->id);
    snprintf(buf, 400, "BN%d", (int) i);
    node_names[to->id] = buf;
    to->set("Threshold", 0);

    all_node_ids.push_back(l->calculate_neuromorphic(all_node_ids[i-1], net, node_names, to, total_sim_time, 0));

    if (from != nullptr) {
        /* all_sim_time[i - 1] - all_sim_time[from_index - 1] is the delay between from bn and to bn
         all_sim_time[i] represents the total_sim_time at layer i after processing layer i. Since the bn for layer i
         always start at the begining of processing that layer. We need to include sim_time of that layer, which is
         why we need to go back one layer (from_index - 1).
        */
        split_edge_delay(net, from->id, to->id, all_sim_time[i - 1] - all_sim_time[from_index - 1], 1, node_names);
    } else {
        head_bn = to;
        net->add_input(head_bn->id);
    }

    from = to;
    from_index = i;

    } else {
    // all_node_ids.push_back(l->calculate_neuromorphic(all_node_ids[i-1], net, node_names, nullptr, total_sim_time, 0));
    }

    all_sim_time.push_back(total_sim_time);

    if (add_layer_info != "") {
      add_layer_info += layers[i-1]->name + ")\n";
      model_summary += std::string(summary_len, '-') + '\n';
      model_summary += add_layer_info;
    }

    model_summary += std::string(summary_len, '-') + '\n';
    l->layer_info.resize(summary_len);
    model_summary += l->layer_info + '\n';

    if (debug_print) cout << "Finish process layer " << l->name << endl;
  }

  model_summary += std::string(summary_len, '=') + '\n';


  snprintf(buf, 400, "Total params: %zu\n", total_params);
  model_summary += (string) buf;

  snprintf(buf, 400, "Total neurons: %zu\nTotal synapses: %zu\nSimulation time: %d\n",
                net->num_nodes(), net->num_edges(), total_sim_time + integration_time);
  model_summary += (string) buf;

  if (integration_time != 0) {
    for (eit = net->edges_begin(); eit != net->edges_end(); ++eit) {
        e = eit->second.get();
        e->set("Delay", e->get("Delay") - integration_time);
    }
  }
  if (debug_print) cout << "Finish process integration_time " << integration_time << endl;

  /* debugging code */
  if (debug_print) {
    set <string> all_names;
    string name;
    string error = "";
    vector <size_t> visited;
    vector <size_t> stack;
    size_t i, j, k;
    Node *n, *to;

    /* make sure each node has a unique name */
    if (node_names.size() != net->num_nodes()) error += "node_names.size() != net->num_nodes()\n";
    for (auto mit = node_names.begin(); mit != node_names.end(); ++mit) {
       name = mit->second;
       if (all_names.find(name) == all_names.end()) all_names.insert(name);
       else error += (name + " is already in the node_names map\n");
    }

    for (auto nit = net->begin(); nit != net->end(); ++nit) {
      n = nit->second.get();
      if (node_names.find(n->id) == node_names.end()) {
        snprintf(buf, 400, "%d is not in the nodes_name map", n->id);
        error += buf;
      }
    }

    /* DFS - make sure all nodes can be reached from input neurons or head bn
       It mainly helps debug the optimized conv2d layer with same padding since I remove
       unreachable R/L neruons there.
    */
    visited.resize(net->num_nodes(), 0);
    for (i = 0; i < A.size(); i++)
      for (j = 0; j < A[0].size(); j++)
        for (k = 0; k < A[0][0].size(); k++) stack.push_back(A[i][j][k]);

    stack.push_back(head_bn->id);
    while (!stack.empty()) {
      n = net->get_node(stack[stack.size() - 1]);
      stack.pop_back();
      if (n->id >= visited.size()) error += "node id skip some numbers\n";
      else if (visited[n->id] != 1) {
        visited[n->id] = 1;
        for (i = 0; i < n->outgoing.size(); i++) {
          to = n->outgoing[i]->to;
          if (visited[to->id] == 0) stack.push_back(to->id);
        }
      }
    }

    if (error != "") throw SRE(error);
    for (i = 0; i < visited.size(); i++) {
      if (visited[i] == 0) {
        snprintf(buf, 400, "Neuron %d(", (int) i);
        error += ((string) buf + node_names[net->get_node(i)->id] +
                 ") can not be reached from input neruons. " +
                 "(All weights going to this neruon may be 0)\n");
      }
    }
    if (error != "") std::cerr << error << endl;
  }

  printf("Head Bias Neuron ID (you will need this to run the network): %d\n", head_bn->id);
}

void Model::write_network(const string &path) {
  json j;
  string s;
  FILE *fp;

  fp = fopen(path.c_str(), "w+");
  if (fp != nullptr) {
    net->to_json(j);
    s = j.dump();
    fprintf(fp, "%s", s.c_str());
  } else {
    printf("Failed to open %s\n", path.c_str());
    return;
  }

  fclose(fp);
}

void Model::load_weights(const json &p_weights) { //take thresholds as input

  int i;

  int pool_size;          /* pool size for max pooling layer */
  int weight_index;       /* current index to json weights */
  int threshold_index;    /* threshold index */
  int zero_out_index;     /* zero_out index */
  int depth;

  Vec4D conv_weights;
  Vec2D dense_weights, softmax_weights;
  Vec1D biases, ts, tmp_zero_out;

  string type;
  string padding;
  string estring;
  bool dense_or_conv;
  bool output_layer;
  int num_layers;
  Layer *l;
  json weights;
  char buf[200];

  weight_index = 0;
  threshold_index = 0;
  zero_out_index = 0;
  total_params = 0;
  num_layers = layer_types.size();

  /* Get rid of non-array element */
  if (!p_weights.is_array()) throw SRE("weights must be an array");
  weights = json::array();
  for (i = 0; i < (int) p_weights.size(); i++) {
    if((depth = check_json_array_depth(p_weights[i])) > 0) {
      if (debug_print) {
        printf("weight json array depth: %d", depth);
        if (depth == 1) printf(" - Biases\n");
        else if (depth == 2) printf(" - Dense layer or softmax weights\n");
        else if (depth == 4) printf(" - Conv2D weights\n");

      }
      weights.push_back(p_weights[i]);
    }
  }
  // zero_out_all_layers(weights, tmp_zero_out, overall_zero_out);
  // if (tmp_zero_out.size() == 0) tmp_zero_out.resize(num_layers + 1, 0);

  /* All neuromorphics will be created will be preprocessed, as to make the input layer its own layer */
  /* Create the input layer
     If we don't preprocess the input layer, we need a threshold for inputs, which is thresholds[0].
     If we don't have enough thresholds, we use previous one.
     If we don't have enough zero_out fraction, we use previous one as well.
  */

  thresholds.resize(num_layers + 1, thresholds[thresholds.size() - 1]);
  l = add_layer(new Input(preprocessing));

  if (!preprocessing) {
    l->threshold = thresholds[0];
    threshold_index++;
  }

  /* create layers based on the layer types */
  try {

    for (i = 0; i < num_layers; i++) {

      dense_or_conv = false;
      if (layer_types[i] != "-") type = layer_types[i];

      ts.clear();
      biases.clear();

      /* check to see if the current layer use biases */
      if ((int) weights.size() > weight_index + 1 &&
          check_json_array_depth(weights[weight_index + 1]) == 1) {
        biases = json_to_vec1d(weights[weight_index + 1]);
      }

      /* When we have softmax decoder layer,
         the layer before softmax decoder is the output layer for neuromorphic network
      */
      if (layer_types[num_layers - 1] == "softmax_decoder") output_layer = (i >= num_layers - 2);
      else output_layer = (i == num_layers - 1);


      if (type == "dense") {

        if (weight_index >= (int) weights.size()) throw SRE("not enough weights for specified layers");
        dense_weights = json_to_vec2d(weights[weight_index]);
        ts.resize(dense_weights[0].size(), thresholds[threshold_index]);

        l = add_layer(new Dense(dense_weights, biases, ts, output_layer));
        l->threshold = thresholds[threshold_index];
        l->weight_shape = VecInt1D { (int) dense_weights.size(),
                                     (int) dense_weights[0].size() };
        l->dense->num_params -= (size_t) (l->dense->num_params);

        total_params += l->dense->num_params;
        dense_or_conv = true;


      }// else if (type.find("conv2d-") == 0) {

      //  if (weight_index >= (int) weights.size()) throw SRE("not enough weights for specified layers");
      //  padding = type.substr(7);
      //  conv_weights = json_to_vec4d(weights[weight_index]);

      //  if (zero_out[zero_out_index] + tmp_zero_out[zero_out_index] > 1) throw SRE("zero_out + overall_zero_out > 1");
      //  zero_out_conv2d(conv_weights, biases, zero_out[zero_out_index]);
      //  ts.resize(conv_weights[0][0][0].size(), thresholds[threshold_index]);
      //  l = add_layer(new Conv2D(conv_weights, biases, ts, padding, output_layer));
      //  l->zero_out = zero_out[zero_out_index] + tmp_zero_out[zero_out_index];
      //  l->threshold = thresholds[threshold_index];
      //  l->weight_shape = VecInt1D { (int) conv_weights.size(),
      //                               (int) conv_weights[0].size(),
      //                               (int) conv_weights[0][0].size(),
      //                               (int) conv_weights[0][0][0].size() };

      //  l->conv2d->num_params -= (size_t) (l->conv2d->num_params * (zero_out[zero_out_index] + tmp_zero_out[zero_out_index]));
      //  total_params += l->conv2d->num_params;
      //  dense_or_conv = true;

      //} else if (type.find("maxpooling-") == 0) {
      //  if (sscanf(type.c_str(), "maxpooling-%d", &pool_size) != 1) throw SRE("maxpolling-poolSize");
      //  l = add_layer(new MaxPooling2D(pool_size, output_layer));

      //} else if (type == "flatten") {
      //  l = add_layer(new Flatten(output_layer));

      //} else if (type == "softmax_decoder") {

      //  if (weight_index >= (int) weights.size()) throw SRE("not enough weights for specified layers");
      //  softmax_weights = json_to_vec2d(weights[weight_index]);

      //  if (zero_out[zero_out_index] + tmp_zero_out[zero_out_index] > 1) throw SRE("zero_out + overall_zero_out > 1");

      //  zero_out_dense_or_softmax(softmax_weights, biases, zero_out[zero_out_index]);

      //  l = add_layer(new SoftmaxDecoder(softmax_weights));
      //  l->zero_out = zero_out[zero_out_index] + tmp_zero_out[zero_out_index];
      //  l->weight_shape = VecInt1D { (int) weights[weight_index].size(),
      //                               (int) weights[weight_index][0].size() };

      //  l->softmax_decoder->num_params -= (size_t) (l->softmax_decoder->num_params * (zero_out[zero_out_index] + tmp_zero_out[zero_out_index]));
      //  total_params += l->softmax_decoder->num_params;
      //  weight_index++;
      //  zero_out_index++;

      //} else {
      //  throw SRE("layertypes's element must be one of dense|conv2d-paddingType|maxpooling-poolSize|flatten|softmax_decoder");
      //}

      if (dense_or_conv) {
        threshold_index++;
        // zero_out_index++;
        weight_index++;
        if (biases.size() != 0) weight_index++;
      }

      if (debug_print) cout << "Finish creating layer " << type << endl;


    }
  } catch (const json::exception &e) {
    snprintf(buf, 200, "(layer %d)", i+1);
    estring = "Model - load_weights(): Errors on " + type + " layer " + buf + ": " + e.what();
    throw SRE(estring);
  } catch (const SRE &e) {
    snprintf(buf, 200, "(layer %d)", i+1);
    estring = "Model - load_weights(): Errors on " + type + " layer " + buf + ": " + e.what();
    throw SRE(estring);
  }


  if (weight_index != (int)weights.size()) throw SRE("Model: weights file does not match to layer_types");

}

Layer *Model::add_layer(Input *input) {
  Layer *l;

  l = new Layer(LayerType::InputLayer);
  l->input = input;
  if (layers.size() != 0) throw SRE("Model - add_layer(): First layer must be the input layer");
  l->set_output_shape(input_shape);
  layers.push_back(l);

  return l;
}

Layer *Model::add_layer(Dense *dense) {
  size_t i;
  int from;
  Layer *pl, *l;
  char buf[256];

  l = new Layer(LayerType::DenseLayer);
  dense->layer_index = layers.size();
  l->dense = dense;

  if (layers.size() == 0) throw SRE("Model - add_layer(): Miss the input layer");

  pl = layers[layers.size() - 1];

  if (pl->layer_type != LayerType::DenseLayer &&
      pl->layer_type != LayerType::FlattenLayer &&
      pl->layer_type != LayerType::InputLayer ) {

    throw SRE("Model - add_layer(): Dense's previous layer is not flatten, dense, or input layer");
  }

  l->set_output_shape(pl->output_shape);

  layers.push_back(l);

  return l;
}

Layer::Layer(const LayerType type) {

  layer_type = type;
  layer_info = "";
  threshold = -1;
  sim_time = -1;
  zero_out = 0;

  // max_pooling2d = nullptr;
  dense = nullptr;
  // conv2d = nullptr;
  // flatten = nullptr;
  // softmax_decoder = nullptr;
  input = nullptr;
}

Layer::~Layer() {
  if (layer_type == LayerType::DenseLayer) delete dense;
  // else if (layer_type == LayerType::Conv2DLayer) delete conv2d;
  // else if (layer_type == LayerType::FlattenLayer) delete flatten;
  // else if (layer_type == LayerType::MaxPooling2DLayer) delete max_pooling2d;
  // else if (layer_type == LayerType::SoftmaxDecoderLayer) delete softmax_decoder;
  else if (layer_type == LayerType::InputLayer) delete input;
}

void Layer::set_divisor(double divisor) {
  if (layer_type == LayerType::DenseLayer) dense->set_divisor(divisor);
  // else if (layer_type == LayerType::Conv2DLayer) conv2d->set_divisor(divisor);
}

Vec3D Layer::calculate_regular(const Vec3D &inputs, bool binary_output, bool collect_layer_info) {
  Vec3D outputs;

  char buf[512];
  char name_buf[100];
  char input_shape_buf[100];
  char output_shape_buf[100];
  char weight_shape_buf[100];
  char threshold_buf[100];

  resize_vec3d(outputs, 1, 1, 0, 0);

  if (binary_output) snprintf(threshold_buf, 100, "%.3lf", threshold);
  else snprintf(threshold_buf, 100, "N/A");

  shape_to_cstr(input_shape_buf, 100, input_shape);
  shape_to_cstr(output_shape_buf, 100, output_shape);
  shape_to_cstr(weight_shape_buf, 100, weight_shape);
  if (layer_type == LayerType::DenseLayer) {
    outputs[0][0] = dense->calculate_regular(inputs[0][0], binary_output);
    snprintf(name_buf, 100, "dense_%d", dense->layer_index);
    snprintf(buf, 512, "%-30s%-20s%-20s%-12s%-12s%-12zu%-12s%-12s%-12.4lf",
                 name_buf, weight_shape_buf, output_shape_buf, "N/A", "N/A", dense->num_params, threshold_buf, "N/A", zero_out);
  } else if (layer_type == LayerType::Conv2DLayer) {
    // outputs = conv2d->calculate_regular(inputs, binary_output);
    // snprintf(name_buf, 100, "conv2d_%s_%d", conv2d->padding.c_str(), conv2d->layer_index);
    // snprintf(buf, 512, "%-30s%-20s%-20s%-12s%-12s%-12zu%-12s%-12s%-12.4lf",
    //              name_buf, weight_shape_buf, output_shape_buf, "N/A", "N/A", conv2d->num_params, threshold_buf, "N/A", zero_out);

  } else if (layer_type == LayerType::MaxPooling2DLayer) {
    // outputs = max_pooling2d->calculate_regular(inputs);
    // snprintf(name_buf, 100, "max_pooling2d_%d_%d", max_pooling2d->pool_size, max_pooling2d->layer_index);
    // snprintf(buf, 512, "%-30s%-20s%-20s%-12s%-12s%-12d%-12s%-12s%-12s",
    //              name_buf, "N/A", output_shape_buf, "N/A", "N/A", 0, "N/A", "N/A", "N/A");

  } else if (layer_type == LayerType::FlattenLayer) {
    // outputs[0][0] = flatten->calculate_regular(inputs);
    // snprintf(name_buf, 100, "flatten_%d", flatten->layer_index);
    // snprintf(buf, 512, "%-30s%-20s%-20s%-12s%-12s%-12d%-12s%-12s%-12s",
    //              name_buf, "N/A", output_shape_buf, "N/A", "N/A", 0, "N/A", "N/A", "N/A");

  } else if (layer_type == LayerType::SoftmaxDecoderLayer) {
    // outputs[0][0] = softmax_decoder->calculate_regular(inputs[0][0]);
    // snprintf(name_buf, 100, "softmax_decoder_%d", softmax_decoder->layer_index);
    // snprintf(buf, 512, "%-30s%-20s%-20s%-12s%-12s%-12zu%-12s%-12s%-12.4lf",
    //              name_buf, weight_shape_buf, output_shape_buf, "N/A", "N/A", softmax_decoder->num_params, "N/A", "N/A", zero_out);

  } else if (layer_type == LayerType::InputLayer) {
    /* do nothing but get layer info */

    snprintf(name_buf, 100, "input_0");
    if (input->preprocessing) {
      snprintf(buf, 512, "%-30s%-20s%-20s%-12s%-12s%-12d%-12s%-12s%-12s",
                   "input_0 (Preprocessing)", "N/A", output_shape_buf, "N/A", "N/A", 0, "N/A", "N/A", "N/A");
    } else {
      snprintf(buf, 512, "%-30s%-20s%-20s%-12s%-12s%-12d%-12s%-12s%-12s",
                   "input_0", "N/A", output_shape_buf, "N/A", "N/A", 0, threshold_buf, "N/A", "N/A");
    }
  }

  /* if threading is used, this has to be false */
  if (collect_layer_info) {
    name = (string) name_buf;
    layer_info = (string) buf;
    snprintf(buf, 512, "[label=\"%s | {input: | weight: | output:} | {%s | %s | %s}\"];\n",
                  name_buf, input_shape_buf, weight_shape_buf, output_shape_buf);
    dot_attr = (string) buf;
  }
  return outputs;
}

VecInt3D Layer::calculate_neuromorphic(const VecInt3D &A,
                                       Network *net,
                                       map <int, string> &node_names,
                                       Node* bn,
                                       int &total_sim_time,
                                       int skip_layer_delay) {
  VecInt3D B;
  char buf[512];
  char name_buf[100];
  char input_shape_buf[100];
  char output_shape_buf[100];
  char weight_shape_buf[100];

  resize_vec_int_3d(B, 1, 1, 0, 0);
  shape_to_cstr(input_shape_buf, 100, input_shape);
  shape_to_cstr(output_shape_buf, 100, output_shape);
  shape_to_cstr(weight_shape_buf, 100, weight_shape);
  if (layer_type == LayerType::DenseLayer) {
    B[0][0] = dense->calculate_neuromorphic(A[0][0], net, node_names, bn, total_sim_time, skip_layer_delay);
    sim_time = 1;
    snprintf(name_buf, 100, "dense_%d", dense->layer_index);
    snprintf(buf, 512, "%-30s%-20s%-20s%-12zu%-12zu%-12zu%-12.3lf%-12d%-12.4lf",
                 name_buf, weight_shape_buf, output_shape_buf, dense->num_neurons, dense->num_synapses, dense->num_params, threshold, 1, zero_out);

  } else if (layer_type == LayerType::Conv2DLayer) {
    // sim_time = total_sim_time;
    // B = conv2d->calculate_neuromorphic(A, net, node_names, bn, total_sim_time, skip_layer_delay);
    // sim_time = total_sim_time - sim_time;
    // snprintf(name_buf, 100, "conv2d_%s_%d (%s)", conv2d->padding.c_str(), conv2d->layer_index, (conv2d->optimal)? "optimal" : "nonoptimal");
    // snprintf(buf, 512, "%-30s%-20s%-20s%-12zu%-12zu%-12zu%-12.3lf%-12d%-12.4lf",
    //              name_buf, weight_shape_buf, output_shape_buf, conv2d->num_neurons, conv2d->num_synapses, conv2d->num_params, threshold, sim_time, zero_out);
    // snprintf(name_buf, 100, "conv2d_%s_%d", conv2d->padding.c_str(), conv2d->layer_index);

  } else if (layer_type == LayerType::MaxPooling2DLayer) {
    // B = max_pooling2d->calculate_neuromorphic(A, net, node_names, total_sim_time);
    // sim_time = 1;
    // snprintf(name_buf, 100, "max_pooling2d_%d_%d", max_pooling2d->pool_size, max_pooling2d->layer_index);
    // snprintf(buf, 512, "%-30s%-20s%-20s%-12zu%-12zu%-12s%-12s%-12d%-12s",
    //              name_buf, "N/A", output_shape_buf, max_pooling2d->num_neurons, max_pooling2d->num_synapses, "N/A", "N/A", 1, "N/A");

  } else if (layer_type == LayerType::FlattenLayer) {
    // B[0][0] = flatten->calculate_neuromorphic(A);
    // snprintf(name_buf, 100, "flatten_%d", flatten->layer_index);
    // snprintf(buf, 512, "%-30s%-20s%-20s%-12d%-12d%-12s%-12s%-12s%-12s",
    //              name_buf, "N/A", output_shape_buf, 0, 0, "N/A", "N/A", "N/A", "N/A");

  } else if (layer_type == LayerType::SoftmaxDecoderLayer) {
    /* do nothing for calculation because softmax decoder only does regular calculation */
    // snprintf(name_buf, 100, "softmax_decoder_%d", softmax_decoder->layer_index);
    // snprintf(buf, 512, "%-30s%-20s%-20s%-12s%-12s%-12zu%-12s%-12s%-12.4lf",
    //              name_buf, weight_shape_buf, output_shape_buf, "N/A", "N/A", softmax_decoder->num_params, "N/A", "N/A", zero_out);
  } else if (layer_type == LayerType::InputLayer) {
    /* do nothing but get layer info */
    snprintf(name_buf, 100, "input_0");

    if (input->preprocessing) {
      snprintf(buf, 512, "%-30s%-20s%-20s%-12s%-12s%-12d%-12s%-12s%-12s",
                   "input_0 (Preprocessing)", "N/A", output_shape_buf, "N/A", "N/A", 0, "N/A", "N/A", "N/A");
    } else {
      snprintf(buf, 512, "%-30s%-20s%-20s%-12zu%-12zu%-12s%-12.3lf%-12s%-12s",
                   "input_0", "N/A", output_shape_buf, net->num_nodes(), net->num_edges(), "N/A", threshold, "N/A", "N/A");
    }
  }

  name = (string) name_buf;
  layer_info = (string) buf;
  snprintf(buf, 512, "[label=\"%s | {input: | weight: | output:} | {%s | %s | %s}\"];\n",
               name_buf, input_shape_buf, weight_shape_buf, output_shape_buf);
  dot_attr = (string) buf;
  return B;
}

void Layer::set_output_shape(const VecInt1D &p_input_shape) {

  input_shape = p_input_shape;
  if (layer_type == LayerType::DenseLayer) output_shape = dense->get_output_shape(input_shape);
  // else if (layer_type == LayerType::Conv2DLayer) output_shape = conv2d->get_output_shape(input_shape);
  // else if (layer_type == LayerType::FlattenLayer) output_shape = flatten->get_output_shape(input_shape);
  // else if (layer_type == LayerType::MaxPooling2DLayer) output_shape = max_pooling2d->get_output_shape(input_shape);
  // else if (layer_type == LayerType::SoftmaxDecoderLayer) output_shape = softmax_decoder->get_output_shape(input_shape);
  else if (layer_type == LayerType::InputLayer) output_shape = input_shape;

}

Input::Input(bool p_preprocessing) {
  preprocessing = p_preprocessing;
}

Dense::Dense(const Vec2D &p_weights, const Vec1D &p_biases, const Vec1D &p_thresholds, bool p_output_layer)
{
  /* init member variables */
  weights = p_weights;
  output_layer = p_output_layer;
  num_params = 0;
  num_neurons = 0;
  num_synapses = 0;
  divisor = -1;
  char buf[200];

  /* If p_biases is empty, we fill biases up with zeros */
  if (p_biases.size() == 0) {
    biases = zeros(weights[0].size());
  } else {
    if (p_biases.size() != weights[0].size()) {
      snprintf(buf, 200, "Dense: #outputs %d != #biases %d", (int) weights[0].size(),
                                                             (int) p_biases.size());
      throw SRE((string) buf);
    }
    biases = p_biases;
    num_params += p_biases.size();
  }

  if (p_thresholds.size() == 0) {
    thresholds = zeros(weights[0].size());
  } else {
    if (p_thresholds.size() != weights[0].size()) {
      snprintf(buf, 200, "Dense: #outputs %d != #thresholds %d",
                    (int) weights[0].size(), (int) p_thresholds.size());
      throw SRE((string) buf);
    }
    thresholds = p_thresholds;
  }

  num_params += weights.size() * weights[0].size();

}

Vec1D Dense::calculate_regular(const Vec1D &inputs, bool binary_output)
{
  size_t row, col;
  double sum;
  Vec1D outputs;
  char buf[200];

  if (inputs.size() != weights.size()) {
    snprintf(buf, 200, "Dense - calculate_regular(): #inputs %d != weights.size() %d.",
             (int) inputs.size(), (int) weights.size());
    throw SRE((string) buf);
  }
  outputs.resize(weights[0].size());

  for (col = 0; col < weights[0].size(); col++) {
    sum = 0;
    for (row = 0; row < weights.size(); row++) {
      sum += weights[row][col] * inputs[row];
    }
    outputs[col] = sum + biases[col];
    if (binary_output) outputs[col] = outputs[col] >= thresholds[col] ? 1 : 0;
  }

  return outputs;
}

VecInt1D Dense::calculate_neuromorphic(const VecInt1D &A,
                                       Network *net,
                                       map <int, string> &node_names,
                                       Node* bn,
                                       int &sim_time,
                                       int skip_layer_delay) {
  int i, j;
  size_t num_neurons_before;            /* number of neurons before adding more */
  size_t num_synapses_before;           /* number of synapses before adding more */
  int wrows, wcols;                     /* rows and cols in the weight matrix */

  double tmp_divisor;                   /* the default divisor */
  bool make_b_outputs;

  Node *n;
  char buf[200];


  num_neurons_before = net->num_nodes();
  num_synapses_before = net->num_edges();

  wrows = weights.size();
  wcols = weights[0].size();



  if ((int) A.size() != wrows) {
    snprintf(buf, 200, "Dense - calculate_neuromorphic(): #inputs %d != #weight matrix's rows %d",
                  (int) A.size(), (int) wrows);
    throw((string) buf);
  }

  /* A non-zero skip_layer_delay is for building the residual block.
     If B has not been created, we scale the weights/thresholds, create B neurons, and BN neurons.
  */
  if (B.size() == 0) {

    /* We can actually get rid of biases value by subtracting thresholds from biases
       Additionally, we want to keep thresholds positive.
       Because if we have negative thresholds, we also need to add a synapses to it with a weight of 0.
    */
    for (i = 0; i < (int) thresholds.size(); i++) {
      if (thresholds[i] - biases[i] > 0) {
        thresholds[i] -= biases[i];
        biases[i] = 0;
      }
    }

    tmp_divisor = get_divisor(net);
    if (divisor != -1 && divisor < tmp_divisor ) {
      fprintf(stderr, "Warning - Dense::calculate_neuromorphic(): divisor %.3lf < default divisor %.3lf.", divisor, tmp_divisor);
      fprintf(stderr, " Some weights and biases wiil be clamped\n");
    }
    if (divisor == -1) divisor = tmp_divisor;

    make_b_outputs = true;
    if (!output_layer) make_b_outputs = !net->get_data("other")["suppress_hidden_outputs"];

    scale_weights_and_thresholds(net);

    for (i = 0; i < wcols; i++) {
      B.push_back(net->num_nodes());
      n = net->add_node(B[i]);
      n->set("Threshold", thresholds[i]);

      if (output_layer) snprintf(buf, 200, "B(%d)", i);
      else snprintf(buf, 200, "H%d(%d)", layer_index, i);
      node_names[B[i]] = buf;
      if (make_b_outputs) net->add_output(B[i]);
    }

    /* if the threshold is negative, we have to send 0 charge in case none of input spikes */
    for (i = 0; i < wcols; i++) {
      if (thresholds[i] <= 0 || biases[i] != 0) {
        add_and_set_edge(net, bn->id, B[i], 1, biases[i]);
      }
    }
  }

  /* fully connect two layers A->B */
  for (i = 0; i < wrows; i++) {
    for (j = 0; j < wcols; j++) {
      if (weights[i][j] != 0) {
        split_edge_delay(net, A[i], B[j], 1 + skip_layer_delay, weights[i][j], node_names);
      }
    }
  }

  /* get # neurons and # synapses for this layer */
  num_neurons += net->num_nodes() - num_neurons_before;
  if (skip_layer_delay == 0) num_neurons++; // 1 for the bias neuron
  num_synapses += net->num_edges() - num_synapses_before;

  sim_time += 1;
  return B;
}

void Dense::set_divisor(double d) {
  if (d <= 0) throw SRE("Dense - set_divisor(): divisor must be > 0");
  divisor = d;
}

VecInt1D Dense::get_output_shape(const VecInt1D &input_shape)
{
  char buf[200];
  if (input_shape.size() != 1) throw SRE("Dense: input shape dim != 1");
  if (input_shape[0] != (int) weights.size()) {
    snprintf(buf, 200,
       "Dense - get_output_shape(): input shape %d != first dimensionality of weight shape %d",
       input_shape[0], (int) weights.size());
       throw SRE((string) buf);
  }
  return VecInt1D {(int) weights[0].size()};
}

double Dense::get_divisor(const Network *net) {
  double max_weight, min_weight;
  double max_threshold, min_threshold;
  double min, max;
  double t_divisor, w_divisor;
  size_t i, j;
  PropertyPack pp;
  const Property *pr;

  pr = net->get_node_property("Threshold");
  max_threshold = pr->max_value;
  min_threshold = pr->min_value;

  pr = net->get_edge_property("Weight");
  max_weight = pr->max_value;
  min_weight = pr->min_value;

  /* find weight divisor */
  min = weights[0][0];
  max = weights[0][0];
  for (i = 0; i < weights.size(); i++) {
    for (j = 0; j < weights[i].size(); j++) {
      if (min > weights[i][j]) min = weights[i][j];
      if (max < weights[i][j]) max = weights[i][j];
    }
  }

  for (i = 0; i < biases.size(); i++) {
    if (min > biases[i]) min = biases[i];
    if (max < biases[i]) max = biases[i];
  }

  w_divisor = -1;
  if (min < 0) w_divisor = min / min_weight;
  if (max > 0 && max / max_weight > w_divisor) w_divisor = max / max_weight;
  if (w_divisor == -1) w_divisor = 1;


  /* find threshold divisor */
  t_divisor = -1;
  min = thresholds[0];
  max = thresholds[0];
  for (i = 0; i < thresholds.size(); i++) {
    if (min > thresholds[i]) min = thresholds[i];
    if (max < thresholds[i]) max = thresholds[i];
  }

  if (min < 0) {
    if (min_threshold >= 0) throw SRE("Dense - get_divisor(): min threshold >= 0 when we have negative threshold");
    t_divisor = min / min_threshold;
  }
  if (max > 0 && max / max_threshold > t_divisor) t_divisor = max / max_threshold;

  return t_divisor > w_divisor ? t_divisor : w_divisor;
}

void Dense::scale_weights_and_thresholds(const Network *net) {
  size_t i, j;
  bool discrete_w, discrete_t;
  double max_weight, min_weight;
  double max_threshold, min_threshold;
  string proc_name;

  const Property *pr;

  if (divisor == -1) throw SRE("Dense - scale_weights_and_thresholds(): divisor is not set");

  discrete_t = (net->get_node_property("Threshold"))->type == Property::Type::INTEGER;
  discrete_w = (net->get_edge_property("Weight"))->type == Property::Type::INTEGER;
  pr = net->get_node_property("Threshold");
  max_threshold = pr->max_value;
  min_threshold = pr->min_value;

  pr = net->get_edge_property("Weight");
  max_weight = pr->max_value;
  min_weight = pr->min_value;

  // proc_name = net->get_data("other")["proc_name"];
  proc_name = "risp";

  /* We allow the weights/thresholds to not be in the range because we clamp them */
  for (i = 0; i < weights.size(); i++) {
    for (j = 0; j < weights[0].size(); j++) {
      weights[i][j] /= divisor;
      weights[i][j] = clamp_value(weights[i][j], min_weight, max_weight, discrete_w);
    }

  }
  for (i = 0; i < biases.size(); i++) {
    biases[i] /= divisor;
    biases[i] = clamp_value(biases[i], min_weight, max_weight, discrete_w);
  }


  for (i = 0; i < thresholds.size(); i++) {
    thresholds[i] /= divisor;
    /* caspian spikes when charge > threshold. So we need to subtract 1 from the threshold. */
    // if (proc_name == "caspian") thresholds[i]--;
    thresholds[i] = clamp_value(thresholds[i], min_threshold, max_threshold, discrete_t);
  }

}
