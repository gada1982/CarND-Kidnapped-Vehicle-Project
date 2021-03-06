/*
 * particle_filter.cpp
 *
 * Created on: May 31, 2017
 * Author: Daniel Gattringer
 * Mail: daniel@gattringer.biz
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // Define the number of particles
  num_particles = 100;
  
  // Create a generator for adding random Gaussian noise
  default_random_engine gen;
  
  // Create a normal (Gaussian) distribution for x with GPS-data as initial mean
  normal_distribution <double> dist_x(x, std[0]);
  
  // Create a normal (Gaussian) distribution for y with GPS-data as initial mean
  normal_distribution <double> dist_y(y, std[1]);
  
  // Create a normal (Gaussian) distribution for theta with GPS-data as initial mean
  normal_distribution <double> dist_theta(theta, std[2]);
  
  // Initialize all particles
  for (int i = 0; i < num_particles; i++) {
    // Create tempory object
    Particle particleObject;
    
    // Set id
    particleObject.id = i;
    
    // Add random Gaussian noise
    particleObject.x = dist_x(gen);
    particleObject.y = dist_y(gen);
    particleObject.theta = dist_theta(gen);
    
    // Set initial weight
    particleObject.weight = 1.0;
    
    // Add object to vector
    particles.push_back(particleObject);
  } // End of for-loop to initialize all particles
  
  // Set init status done
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // Create a generator for adding random Gaussian noise
  default_random_engine gen;
  
  // Creates a normal (Gaussian) distribution for x with GPS as mean
  normal_distribution<double> dist_x(0, std_pos[0]);
  
  // Creates a normal (Gaussian) distribution for y with GPS as mean
  normal_distribution<double> dist_y(0, std_pos[1]);
  
  // Creates a normal (Gaussian) distribution for theta with GPS as mean
  normal_distribution<double> dist_theta(0, std_pos[2]);
  
  // Calculate sub-terms
  const double v_diff_yaw_rate = velocity / yaw_rate;
  const double yaw_rate_mul_delta_t = yaw_rate * delta_t;
  const double v_mul_delta_t = velocity * delta_t;
  
  // Prediction for all particles
  for (int i = 0; i < num_particles; i++) {
    // Get _0 data from a single particle
    const double x_0 = particles[i].x;
    const double y_0 = particles[i].y;
    const double theta_0 = particles[i].theta;
    double x_f = 0;
    double y_f = 0;
    double theta_f = 0;
    
    // Calculate _f data for a single particle (check if car is going straight)
    if (fabs(yaw_rate) > 0.00001) {
      x_f = x_0 + v_diff_yaw_rate*(sin(theta_0 + yaw_rate_mul_delta_t) - sin(theta_0));
      y_f = y_0 + v_diff_yaw_rate*(cos(theta_0) - cos(theta_0 + yaw_rate_mul_delta_t));
      theta_f = theta_0 + yaw_rate_mul_delta_t;
    }
    else {
      x_f = x_0 + v_mul_delta_t*sin(theta_0);
      y_f = y_0 + v_mul_delta_t*cos(theta_0);
      // Theta is not changed -> car doesn't steer
    }
    
    // Add random Gaussian noise
    particles[i].x = x_f + dist_x(gen);
    particles[i].y = y_f + dist_y(gen);
    particles[i].theta = theta_f + dist_theta(gen);
  } // End of for-loop for prediction of all particles
  
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  int closest_id;
  double min_dist;
  double dist_pred_obs;
  
  // Find the closest landmark to each observation
  for (int i = 0; i < observations.size(); i++) {
    min_dist = 1000000; // High value for initialization
    
    for (int j = 0; j < predicted.size(); j++) {
      // Calculate distance between observation and prediction
      dist_pred_obs = dist(predicted[j].x, predicted[j].y, observations[i].x, observations[i].y);
      
      // Set id if smallest distance until now
      if(dist_pred_obs < min_dist)
      {
        min_dist = dist_pred_obs;
        closest_id = j;
      }
    } // End of inner for-loop (predictions)
    
    // Set observation id to the id of the closest prediction
    observations[i].id = closest_id;
  } // End of outer for-loop (oberservations)
  
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations, Map map_landmarks) {
  // Set standard deviation in x and y direction --> description in particle_filter.h is wrong (not allowed to change)
  const double std_x = std_landmark[0];
  const double std_y = std_landmark[1];
  
  // Precalculate const diff-terms
  const double mult_term_x_y = 1.0 / (2.0 * M_PI * std_x * std_y);
  const double diff_term_x = 2.0 * pow(std_x, 2);
  const double diff_term_y = 2.0 * pow(std_y, 2);
  
  // Update weights for each particle
  for (int i = 0; i < num_particles; i++) {
    // Get the data for a single particle
    const double part_x = particles[i].x;
    const double part_y = particles[i].y;
    const double part_theta = particles[i].theta;
    
    // Create a vector of landmarks which are in sensor-range
    vector<LandmarkObs> lm_obs_sensor_range;
    for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      // Create tempory object
      LandmarkObs temp_LandmarkOb;
      
      temp_LandmarkOb.id = map_landmarks.landmark_list[j].id_i;
      temp_LandmarkOb.x = map_landmarks.landmark_list[j].x_f;
      temp_LandmarkOb.y = map_landmarks.landmark_list[j].y_f;
      
      const double dist_lm_part = dist(temp_LandmarkOb.x, temp_LandmarkOb.y, part_x, part_y);
      
      // Only add to vector if in sensor range
      if(dist_lm_part <= sensor_range) {
        lm_obs_sensor_range.push_back(temp_LandmarkOb);
      }
    }  // End of inner for-loop landmark in sensor range
    
    // Transform observations from vehicle's coordinate system into map's coordinate system.
    vector<LandmarkObs> observations_map_coord;
    for (int j = 0; j < observations.size(); j++) {
      // Create tempory object
      LandmarkObs temp_Obs;
      
      const double obs_x = observations[j].x;
      const double obs_y = observations[j].y;
      
      temp_Obs.id = observations[j].id;
      temp_Obs.x = obs_x*cos(part_theta) - obs_y*sin(part_theta) + part_x;
      temp_Obs.y = obs_x*sin(part_theta) + obs_y*cos(part_theta) + part_y;
      
      observations_map_coord.push_back(temp_Obs);
    } // End of inner for-loop Transform coordinate system
    
    // Apply association between transformed observations an landmarks within sensor range
    dataAssociation(lm_obs_sensor_range, observations_map_coord);
    
    // Calculate the weight of each single particle
    // Set particle-weight to initial 1.0
    particles[i].weight = 1.0;
    for (int j = 0; j < observations_map_coord.size(); j++) {
      // Get the index for the observations
      const int j_obs = observations_map_coord[j].id;
      
      // Calculate sub-terms
      const double calc_x = pow(lm_obs_sensor_range[j_obs].x - observations_map_coord[j].x, 2);
      const double calc_y = pow(lm_obs_sensor_range[j_obs].y - observations_map_coord[j].y, 2);
      const double calc_exp = calc_x/diff_term_x + calc_y/diff_term_y;
      
      // Calculate weight from a single observation
      const double calc_single_weight = mult_term_x_y * exp(-1 * calc_exp);
      
      // Calculate particles final weight
      particles[i].weight *= calc_single_weight;
      
    } // End of inner for-loop calculation of particles final weight out of the weight from a single observation
    
  } // End of outer for-loop to update weights for all particle
  
}

void ParticleFilter::resample() {
  // Resample particles with replacement with probability proportional to their weight.
  // NOTE: Details about std::discrete_distribution can be found here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  
  // Create a random number generator
  random_device rd;
  mt19937 gen(rd());
  
  // Get a vector with all the weights
  vector<double> vec_weights;
  for (int i = 0; i < num_particles; i++) {
    vec_weights.push_back(particles[i].weight);
  }
  
  // Create a distribution
  discrete_distribution<> distr(vec_weights.begin(), vec_weights.end());
  
  // Generate re-sampled particles
  vector<Particle> resampled_particles;
  for (int i = 0; i < num_particles; i++) {
    // Create tempory object
    Particle temp_Particle;
    
    // Choose a new particle according to the distribution
    const int i_new = distr(gen);
    temp_Particle = particles[i_new];
    
    resampled_particles.push_back(temp_Particle);
  } // End for-loop Transform coordinate system
  
  // Set the updated particles
  particles = resampled_particles;
}


Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  
  //Clear the previous associations
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();
  
  particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;
  
 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
  vector<int> v = best.associations;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseX(Particle best)
{
  vector<double> v = best.sense_x;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseY(Particle best)
{
  vector<double> v = best.sense_y;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
