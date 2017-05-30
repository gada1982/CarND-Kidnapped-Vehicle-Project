/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
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
  // Define the number of particles and create an array with this size.
  num_particles = 100;
  
  // Create a generator for adding random Gaussian noise
  default_random_engine gen;

  // Creates a normal (Gaussian) distribution for x with GPS as mean
  normal_distribution <double> dist_x(x, std[0]);
  
  // Creates a normal (Gaussian) distribution for y with GPS as mean
  normal_distribution <double> dist_y(y, std[1]);
  
  // Creates a normal (Gaussian) distribution for theta with GPS as mean
  normal_distribution <double> dist_theta(theta, std[2]);
  
  // Initialize all particles
  for (int i = 0; i < num_particles; i++)
  {
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
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // Create a generator for adding random Gaussian noise
  default_random_engine gen;
  
  // Creates a normal (Gaussian) distribution for x with GPS as mean
  normal_distribution <double> dist_x(0, std_pos[0]);
  
  // Creates a normal (Gaussian) distribution for y with GPS as mean
  normal_distribution <double> dist_y(0, std_pos[1]);
  
  // Creates a normal (Gaussian) distribution for theta with GPS as mean
  normal_distribution <double> dist_theta(0, std_pos[2]);
  
  // Calculate sub-terms
  const double v_diff_yaw_rate = velocity / yaw_rate;
  const double yaw_rate_mul_delta_t = yaw_rate * delta_t;
  
  // Initialize all particles
  for (int i = 0; i < num_particles; i++)
  {
    // Get _0 data from a single particle
    const double x_0 = particles[i].x;
    const double y_0 = particles[i].y;
    const double theta_0 = particles[i].theta;
    
    // Calculate _f data for a single particle
    const double x_f = x_0 + v_diff_yaw_rate*(sin(theta_0 + yaw_rate_mul_delta_t) - cos(theta_0));
    const double y_f = y_0 + v_diff_yaw_rate*(cos(theta_0) - cos(theta_0 + yaw_rate_mul_delta_t));
    const double theta_f = theta_0 + yaw_rate_mul_delta_t;
    
    // Add random Gaussian noise
    particles[i].x = x_f + dist_x(gen);
    particles[i].y = y_f + dist_y(gen);
    particles[i].theta = theta_f + dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  int closest_id;
  double min_dist;
  double dist_pred_obs;
  
  for (int i = 0; i < observations.size(); i++)
  {
    min_dist = 1000000; // High value for initialization
    
    for (int j = 0; j < predicted.size(); j++)
    {
      // Calculate distance between observation and prediction
      dist_pred_obs = dist(predicted[j].x, predicted[j].y, observations[i].x, observations[i].y);
      
      // Set id if smallest distance until now
      if(dist_pred_obs < min_dist)
      {
        min_dist = dist_pred_obs;
        closest_id = j;
      }
    }
    
    // Set observation id to the id of the closest prediction
    observations[i].id = closest_id;
  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

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
