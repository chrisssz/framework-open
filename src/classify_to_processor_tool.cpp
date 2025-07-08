#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstdio>
#include <unistd.h>
#include <set>

#include "framework.hpp"
#include "utils/json_helpers.hpp"

using namespace std;
using nlohmann::json;

typedef std::runtime_error SRE;

#define CLASSIFY_TIMESERIES 0b11
#define CLASSIFY_DENSE 0b1
#define CLASSIFY_SPARSE 0b10
#define CLASSIFY_REGULAR 0b100
#define CLASSIFY_UNDEFINE 0b1000

class Observation {
public:
  vector < vector<double> > data;
  vector < vector<double> > time;
  vector <double> labels;
};

void create_command (string &command,
                     const vector < Observation* > data_obs,
                     const double &threshold,
                     const int &sim_time,
                     const int &head_bn_id);

void read_csv_files(vector < Observation* > &data_obs,
                    const string &data_csv,
                    const string &labels_csv,
                    json &features,
                    const set < string > &json_categories,
                    vector < string > &categories);

static inline string toString(const double d) {
  std::ostringstream oss;
  oss << d;
  return oss.str();
}

static inline bool isStringDigits(const string &s, bool include_dot = false) {
  size_t i;
  for (i = 0; i < s.size(); i++) {
    if (include_dot && s[i] == '.') {}
    else if (!(s[i] >= '0' && s[i] <= '9')) return false;
  }
  return true;
}

void print_commands(FILE *f) {
  fprintf(f, "This is a processor_tool command generator. The commands listed below are case-insensitive,\n");
  fprintf(f, "For commands that take a json either put a filename on the same line,\n");
  fprintf(f, "or the json can be multiple lines, starting on the next line.\n\n");


  fprintf(f, "Network Loading Commands\n");
  fprintf(f, "Action commands --\n");
  fprintf(f, "PROC_NAME/PN proc_name               - Set the processor name\n");
  fprintf(f, "PROC_PARAMS/PP proc_param_json       - Load the processor params\n");
  fprintf(f, "NETWORK/N network_json               - Load the network\n");
  fprintf(f, "THRESHOLD/T threshold                - Threshold for the input layer\n");
  fprintf(f, "SIM_TIME sim_time                    - Simulation time for the network (must be >= # layers in the network)\n");
  fprintf(f, "HEAD_BN_ID/HBN head_bias_neuron_id   - Id value of the head bias neuron (should be output during network conversion\n");
  fprintf(f, "BUILD                                - Create the processor_tool command\n");
  fprintf(f, "WRITE filename                       - Save the processor tool commands to a specified file\n");

  fprintf(f, "Classify Commands\n");
  fprintf(f, "LOAD/L data label                    - Load data and labels (only regular timeseries is supported)\n");
  fprintf(f, "FEATURES                             - Data features, can be obtained from the classify_tool\n");
  fprintf(f, "CATEGORIES                           - Json of label categories, must match labels in labels file\n");
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

  string proc_name, proc_params_file, data_path;
  string prompt, cmd;
  string l,s;
  string command;
  FILE *f;
  double threshold;
  int sim_time, head_bn_id;
  size_t i;
  bool isBuilt;

  vector <Observation*> data_obs;
  vector <string> categories;
  vector <string> layer_types;

  set <string> json_categories;

  json proc_params, network_json, features_json, categories_json;

  ofstream fout;

  istringstream ss;

  vector <string> sv;

  if (argc > 2 || (argc == 2 && strcmp(argv[1], "--help") == 0)) {
  fprintf(stderr, "usage: classify_to_processor_tool [prompt]\n");
  fprintf(stderr, "\n");
  print_commands(stderr);
  exit(1);
  }

  if (argc == 2) {
  prompt = argv[1];
  prompt += " ";
  }

  proc_name.clear();
  proc_params.clear();
  network_json.clear();
  data_path.clear();
  threshold = 0.5;
  sim_time = -1;
  head_bn_id = -1;
  isBuilt = false;

  while(1) {
    try {
      if (prompt != "") printf("%s", prompt.c_str());
      if (!getline(cin, l)) exit(0);
      sv.clear();
      ss.clear();
      ss.str(l);

      while (ss >> s) sv.push_back(s);

      if (sv.size() != 0) to_uppercase(sv[0]);

      if (sv.size() == 0 || sv[0][0] == '#') {
      } else if (sv[0] == "?") {
      print_commands(stdout);
      } else if (sv[0] == "Q") {
      exit(0);
      } else if (sv[0] == "PROC_NAME" || sv[0] == "PN") {
        if (sv.size() != 2) printf("usage: PROC_NAME processor_name\n");
        else proc_name = sv[1];
      } else if (sv[0] == "PROC_PARAMS" || sv[0] == "PP") {
        if (!read_json(sv, 1, proc_params)) printf("usage: PROC_PARAMS/PP proc_param_json: Bad json\n");
      } else if (sv[0] == "NETWORK" || sv[0] == "N") {
        if (!read_json(sv, 1, network_json)) printf("usage: NETWORK/N network_json: Bad json\n");
      } else if (sv[0] == "THRESHOLD" || sv[0] == "T") {
        if (sscanf(sv[1].c_str(), "%lf", &threshold) != 1) {
          printf("usage: THRESHOLD/T threshold: Bad value %s\n", sv[1].c_str());
        }
      } else if (sv[0] == "SIM_TIME") {
        if (sscanf(sv[1].c_str(), "%d", &sim_time) != 1) {
          printf("usage: SIM_TIME sim_time: Bad value %s\n", sv[1].c_str());
        }
      } else if (sv[0] == "HEAD_BN_ID" || sv[0] == "HBN") {
        if (sscanf(sv[1].c_str(), "%d", &head_bn_id) != 1) printf("usage: HEAD_BN_ID/HBN head_bias_neuron_id: Bad Value %s\n", sv[1].c_str());
      } else if (sv[0] == "FEATURES") {
        if (!read_json(sv, 1, features_json)) printf("usage: FEATURES features_json: Bad json\n");
      } else if (sv[0] == "CATEGORIES") {
        if (!read_json(sv, 1, categories_json)) printf("usage: CATGEORIES categroies_json: Bad json\n");
        else {

          for (i = 0; i < categories_json.size(); i++) {

            if (categories_json[i].is_string()) {
              s = categories_json[i];
            } else if(categories_json[i].is_number()) {
              s = toString(categories_json[i].get<double>());
            } else {
              throw SRE("categories value must be number or string type");
            }
            if (json_categories.find(s) != json_categories.end()) {
              throw SRE((string) "duplicated category: " + s);
            }
            json_categories.insert(s);
          }
        }
      } else if (sv[0] == "LOAD" || sv[0] == "L") {
        if (sv.size() != 3) {
          printf("usage: LOAD/L data label\n");
        }
        else if (features_json.empty()) printf("Must set FEATURES before calling LOAD\n");
        else if (categories_json.empty()) printf("Must set CATEGORIES before calling LOAD\n");
        else {
          for (i = 0; i < data_obs.size(); i++) delete data_obs[i];
          data_obs.clear();
          categories.clear();
          read_csv_files(data_obs, sv[1], sv[2], features_json, categories_json, categories);
        }
      } else if (sv[0] == "BUILD") {
        if (proc_name.empty() || proc_params.empty() || network_json.empty()) {
          printf("Must set PROC_NAME, PROC_PARAMS, and NETWORK before running BUILD\n");
        } else if (sim_time < 0) {
          printf("SIM_TIME must be greater than 0\n");
        } else if (head_bn_id < 0) {
          printf("Must set HEAD_BN_ID before running BUILD\n");
        } else if (data_obs.empty()) {
          printf("Must LOAD classify data before running BUILD\n");
        } else {

          command.clear();
          command = "MAKE " + proc_name + "\n" + proc_params.dump() + "\n";
          command += "LOAD \n" + network_json.dump() + "\n";

          create_command(command, data_obs, threshold, sim_time, head_bn_id);
          isBuilt = true;
        }
      } else if (sv[0] == "WRITE") {
        if (!isBuilt) printf("Must run BUILD before WRITE\n");
        if (sv.size() != 2) printf("usage: WRITE filename\n");
        else {
          f = fopen(sv[1].c_str(), "w+");
          fprintf(f, "%s", command.c_str());
          fclose(f);
        }
      } else {
        printf("Invalid command %s.  Use '?' to print a list of commands.\n", sv[0].c_str());
      }
    } catch (const std::exception &e) {
      printf("%s\n", e.what());
    }
  }
}

void create_command (string &command,
                     const vector < Observation* > data_obs,
                     const double &threshold,
                     const int &sim_time,
                     const int &head_bn_id)
{
  string cmd_str, buf;
  int current_node_id;
  size_t i, j, k;

  for (i = 0; i < data_obs.size(); i++) {
    command += "CA\n";
    for (j = 0; j < data_obs[i]->data.size(); j++) {
      for (k = 0; k < data_obs[i]->data[j].size(); k++) {
        if (data_obs[i]->data[j][k] >= threshold) {
          command += "AS " + to_string(current_node_id) + " 0 1\n";
        }
        current_node_id++;
      }
    }

    command += "AS " + to_string(head_bn_id) + " 0 1\n";
    command += "RUN " + to_string(sim_time) + "\n";
    command += "OC\n";
    current_node_id = 0;
  }
}

void read_csv_files(vector < Observation* > &data_obs,
                    const string &data_csv,
                    const string &labels_csv,
                    json &features,
                    const set < string > &json_categories,
                    vector < string > &categories)
{
  fstream in;
  int fpo;
  istringstream ss, val_get;
  string line, dline, label, s;
  Observation *o;
  int index;
  json j1;
  double sparse_max;
  int num_predictions = 1;

  unsigned char timeseries = CLASSIFY_REGULAR;

  map <string, int>::const_iterator it;
  set <string> labels_set;     /* used if labels are string */

  /* used if all labels are int.  Why do we have this?  If we treat everything as string when all
  of labels are actually ints, one of issues is that string comparison is different than int
  comparison("2" is greater than "10"). We want to sort ints in this case.  */

  set <int> labels_int_set;

  set <string>::const_iterator sit;
  set <int>::const_iterator int_sit;

  vector <string> labels_vec;   /* labels_vec[i] holds the label for observation i. */
  map <string, int> labels_to_int;

  vector <Observation*> obs;

  double val, t;
  size_t i, j, k;
  string dt;
  bool is_int_labels;
  char buf[200];
  string estring;
  int line_no;
  vector <string> sv;

  /* read data file */
  in.open(data_csv, ios::in);
  if (!in.is_open()) throw SRE("Can't open data csv file " + data_csv);

  o = NULL;
  if (timeseries & CLASSIFY_DENSE) index = 0;
  line_no = 0;
  sparse_max = 0;

  while (getline(in, line)) {
    line_no++;
    sv.clear();
    ss.clear();
    ss.str(line);
    while (getline(ss, dline, ',')) sv.push_back(dline);

    /* When you read a blank line, then:
       - If it's sparse timeseries, then the blank line ends the features for that point.
       - If it's dense timeseries, then it should denote the end of the features for
         that data set.
     */

    if (sv.size() == 0) {
      if (timeseries & CLASSIFY_SPARSE) {
        if (o == NULL) {
          o = new Observation;
          o->data.resize(features.size());
          o->time.resize(features.size());
          obs.push_back(o);
        }
      }
      if (timeseries & CLASSIFY_DENSE) {
        if (index != 0) {
          snprintf(buf, 200, "Error reading line %d - blank line after feature %d.\n",
                   line_no, index);
          estring = buf;
          snprintf(buf, 200, "Blank lines should come after feature: %d.", (int) features.size()-1);
          throw SRE(estring + buf);
        }
      }

      o = NULL;

    } else {

      if (timeseries & CLASSIFY_SPARSE) {
        if (sv.size() % 3 != 0) {
          snprintf(buf, 200, "Error reading line %d - values should be feature,val,time", line_no);
          throw SRE((string)buf);
        }

        for (j = 0; j < sv.size(); j += 3) {
          if (sscanf(sv[j].c_str(), "%d", &index) == 0) {
            snprintf(buf, 200, "Error reading line %d - values should be feature,val,time (%s)",
                     line_no, sv[j].c_str());
            throw SRE((string)buf);
          }
          if (index < 0 || index >= (int) features.size()) {
            snprintf(buf, 200, "Error reading line %d - bad feature number (%d)", line_no, index);
            throw SRE((string)buf);
          }

          if (sscanf(sv[j+1].c_str(), "%lg", &val) == 0) {
            snprintf(buf, 200, "Error reading line %d - values should be feature,val,time (%s)",
                    line_no, sv[j+1].c_str());
            throw SRE((string)buf);
          }
          if (sscanf(sv[j+2].c_str(), "%lg", &t) == 0) {
            snprintf(buf, 200, "Error reading line %d - values should be feature,val,time (%s)",
                    line_no, sv[j+2].c_str());
            throw SRE((string)buf);
          }

          // if (sparse_interval != -1 && t+1 > sparse_interval) {
          //   snprintf(buf, 200, "Error reading line %d: time (%lg) is too big. sparse_interval=%lg.",
          //           line_no, t, sparse_interval);
          //   throw SRE((string)buf);
          // }

          if (o == NULL) {
            o = new Observation;
            o->data.resize(features.size());
            o->time.resize(features.size());
            obs.push_back(o);
          }
          o->data[index].push_back(val);
          o->time[index].push_back(t);
          if (t+1 > sparse_max) sparse_max = t+1;
        }

      } else if (timeseries & CLASSIFY_DENSE) {

        if (index == 0) {
          o = new Observation;
          o->data.resize(features.size());
          obs.push_back(o);
        }

        for (j = 0; j < sv.size(); j++) {
          if (sscanf(sv[j].c_str(), "%lg", &val) == 0) {
            snprintf(buf, 200, "Error reading line %d - non-numeric value (%s)",
                     line_no, sv[j].c_str());
            throw SRE((string)buf);
          }
          o->data[index].push_back(val);
        }
        index++;
        index %= features.size();

      } else {                          // Not dense or sparse
        if (features.size() == 0) {
          for (i = 0; i < sv.size(); i++) features.push_back(json::object());
        }
        if (sv.size() != features.size()) {
          snprintf(buf, 200, "Error reading line %d - wrong # of features (%d)",
                   line_no, (int) sv.size());
          throw SRE((string)buf);
        }
        o = new Observation;
        o->data.resize(1);
        obs.push_back(o);
        for (j = 0; j < sv.size(); j++) {
          if (sscanf(sv[j].c_str(), "%lg", &val) == 0) {
            snprintf(buf, 200, "Error reading line %d - non-numeric value (%s)",
                   line_no, sv[j].c_str());
            throw SRE((string)buf);
          }
          o->data[0].push_back(val);
        }
      }
    }
  }

  in.close();

  /* This sucks, but I don't really have the energy to fix it.  If you're doing sparse,
     and the last value for a feature is not on the last timestep, then
     you need to put an extra time into the times array to denote the length of the
     interval.

     Please read the spike encoder markdown, starting with "You may
     also specify timeseries values that arrive at non-uniform times".  That explains it.

     Here is where we go ahead and put that extra time in.
   */

  // if (timeseries & CLASSIFY_SPARSE) {
  //   if (sparse_interval == -1) sparse_interval = sparse_max;
  //   for (i = 0; i < obs.size(); i++) {
  //     o = obs[i];
  //     for (j = 0; j < o->time.size(); j++) {
  //       if (o->time[j].size() != 0) {
  //         last_time = o->time[j][o->time[j].size()-1];
  //         if (sparse_interval-last_time > 1) o->time[j].push_back(sparse_interval);
  //       }
  //     }
  //   }
  // }

  if (obs.size() == 0) throw SRE("Data file does not contain any data");

  /* With dense timeseries, each feature needs to have the same number of data points,
     so double-check that here. */

  if (timeseries & CLASSIFY_DENSE) {
    fpo = obs[0]->data[0].size();
    for (i = 0; i < obs.size(); i++) {
      o = obs[i];
      if (o->data.size() != obs[0]->data.size()) {
        snprintf(buf, 200, "Observation 0 has %d feature%s and observation %d has %d.",
                 (int) obs[0]->data.size(), (obs[0]->data.size() == 1) ? "" : "s",
                 (int) i, (int) o->data.size());
        throw SRE((string) buf);
      }
      for (j = 0; j < o->data.size(); j++) {
        if ((int) o->data[j].size() != fpo) {
          snprintf(buf, 200,
                   "%s 0 feature 0 has %d data point%s per feature.  %s %d, feature %d  has %d.",
                   "Observation", fpo, ((fpo == 1) ? "" : "s"), "Observation",
                   (int) i, (int) j, (int) o->data[j].size());
          throw SRE((string) buf);
        }
      }
    }
  }

  /* read label file

     When we store the observations, we only store a label number, rather than the
     label string.  So we need to convert strings to numbers.  We use two sets for
     this: labels_set and labels_int_set.  While all of the labels are integers,
     we put the labels into both sets.  When we determine any label isn't a number,
     then we stop using labels_int_set.

     If the categories have been defined in the JSON, then they are in the json_categories
     set.  So go through the target set and set up the categories.
  */

  is_int_labels = true;
  if (json_categories.size() > 0) {
    for (sit = json_categories.begin(); sit != json_categories.end(); sit++) {
      label = *sit;
      labels_set.insert(label);
      if (isStringDigits(label) == false) is_int_labels = false;
      if (is_int_labels) labels_int_set.insert(stoi(label));
    }
  }


  /* Now read the labels.  If we've already set up json_categories, then double-check the
    json_categories set for the label.  If we haven't, then use each label to help define
    labels_set/labels_int_set.
   */

  in.open(labels_csv, ios::in);
  if (!in.is_open()) throw SRE("Can't open the label csv file " + labels_csv);

  while(getline(in, line, ',')) {
    ss.clear();
    ss.str(line);
    while (ss >> label) {
      labels_vec.push_back(label);
      if (json_categories.size() == 0) {
        labels_set.insert(label);
        if (isStringDigits(label) == false) is_int_labels = false;
        if (is_int_labels) labels_int_set.insert(stoi(label));
      } else {
        if (json_categories.find(label) == json_categories.end()) {
          throw SRE((string) "Categories are specified in the JSON, and\n" +
                    "the following label is in the label file, but not in the categories: " +
                    label);
        }
      }
    }
  }
  in.close();

  /* Double-check that the data and labels have the same number of elements. */

  if (obs.size() * num_predictions != labels_vec.size()) {
    throw SRE("number of labels > number of data points.");
  }

  /* When this is regression problem. We don't need to convert labels, so skip
     this step.  Otherwise, we are going to create the categories vector from the
     labels in their proper order. */

  if (true) {

    categories.clear();

    if (is_int_labels) {
      for (int_sit = labels_int_set.begin(); int_sit != labels_int_set.end(); ++int_sit) {
        categories.push_back(toString(*int_sit));
      }
    } else {
      for (sit = labels_set.begin(); sit != labels_set.end(); ++sit) categories.push_back(*sit);
    }

    /* Go through the categories in order, and make sure that labels_to_int is set for
       the categories, so that it's easy to get from categories to integer. */

    for (i = 0; i < categories.size(); i++) {
      label = categories[i];
      if (json_categories.size() == 0 || json_categories.find(label) != json_categories.end()) {
        labels_to_int[label] = i;
      } else {
        fprintf(stderr, "Internal Error -- label not in json_categories: %s\n",
                label.c_str());
        exit(1);
      }
    }
  }

  /* Now, go through the observations, and set the label number for each observation. */

  for (i = 0; i < obs.size(); i++) {
    if (true) obs[i]->labels.push_back(labels_to_int[labels_vec[i]]);
    else {
      for (j = 0; j < (size_t) num_predictions; j++) {
        if (isStringDigits(labels_vec[i * num_predictions + j], true) == false) {
          throw SRE(labels_vec[i * num_predictions + j] + " is not a number");
        }
        obs[i]->labels.push_back( stod(labels_vec[i * num_predictions + j]) );
      }
    }
  }

  /* If the feature information has not been specified, then create it.
     Sparse datasets automatically
     have a minimum of zero, since that's the "value" when there's nothing. */

  if (features.size() == 0) {
    throw SRE("Internal error -- features.size() == 0 in read_csv_file().  Shouldn't happen.");
  }

  if (features[0].size() == 0) {

    for (i = 0; i < features.size(); i++) {
      features[i] = json::object();
      features[i]["low"] = (timeseries & CLASSIFY_SPARSE) ? 0 : 0xfffffff;
      features[i]["high"] = -0xfffffff;
      features[i]["type"] = "D";
    }

    for (i = 0; i < obs.size(); i++) {
      o = obs[i];

      for (j = 0; j < o->data.size(); j++) {
        for (k = 0; k < o->data[j].size(); k++) {
          val = o->data[j][k];

          if (timeseries & CLASSIFY_TIMESERIES) {
            if (val < features[j]["low"].get<double>()) features[j]["low"] = val;
            if (val > features[j]["high"].get<double>()) features[j]["high"] = val;
          } else {
            if (val < features[k]["low"].get<double>()) features[k]["low"] = val;
            if (val > features[k]["high"].get<double>()) features[k]["high"] = val;
          }
        }
      }
    }

    /* If a feature has not been set, then it will automatically get a low of
       zero and a high of one. */

    for (i = 0; i < features.size(); i++) {
      if (features[i]["high"] == -0xfffffff) {
        features[i]["low"] = 0;
        features[i]["high"] = 1;
      }
    }
  }

  /* Go ahead and set data_obs to the observations.  We'll mess with this later (e.g.
     split it into training/testing), depending on what we're doing. */

  data_obs = obs;
}
