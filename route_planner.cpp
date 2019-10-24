#include "route_planner.h"
#include <algorithm>
#include <vector>
#include <iostream>
using std::sort;
using std::vector;
using std::reverse;
using std::cout;
RoutePlanner::RoutePlanner(RouteModel &model, float start_x, float start_y, float end_x, float end_y): m_Model(model) {
    // Convert inputs to percentage:
    start_x *= 0.01;
    start_y *= 0.01;
    end_x *= 0.01;
    end_y *= 0.01;

    // TODO 2: Use the m_Model.FindClosestNode method to find the closest nodes to the starting and ending coordinates.
    // Store the nodes you find in the RoutePlanner's start_node and end_node attributes.
 this->start_node = & m_Model.FindClosestNode(start_x, start_y);
this->end_node = & m_Model.FindClosestNode(end_x, end_y);

}


// TODO 3: Implement the CalculateHValue method.
// Tips:
// - You can use the distance to the end_node for the h value.
// - Node objects have a distance method to determine the distance to another node.

float RoutePlanner::CalculateHValue(RouteModel::Node const *node) {
float dist = node->distance(*this->end_node);
return dist;
}


// TODO 4: Complete the AddNeighbors method to expand the current node by adding all unvisited neighbors to the open list.
// Tips:
// - Use the FindNeighbors() method of the current_node to populate current_node.neighbors vector with all the neighbors.
// - For each node in current_node.neighbors, set the parent, the h_value, the g_value. 
// - Use CalculateHValue below to implement the h-Value calculation.
// - For each node in current_node.neighbors, add the neighbor to open_list and set the node's visited attribute to true.

void RoutePlanner::AddNeighbors(RouteModel::Node *current_node) {
current_node->FindNeighbors();
for (RouteModel::Node *c_node : current_node->neighbors) {
    if (c_node->visited == false){
    c_node->parent = current_node;
    c_node->h_value = CalculateHValue(c_node);
    c_node->g_value = c_node->distance(*current_node) + current_node->g_value;
    c_node->visited = true;
    this->open_list.emplace_back(c_node);}}
//current_node->visited = true;

}



// TODO 5: Complete the NextNode method to sort the open list and return the next node.
// Tips:
// - Sort the open_list according to the sum of the h value and g value.
// - Create a pointer to the node in the list with the lowest sum.
// - Remove that node from the open_list.
// - Return the pointer.

bool Compare(RouteModel::Node * n1, RouteModel::Node * n2) {
  int f1 = n1->g_value + n1->h_value; // f1 = g1 + h1
  int f2 = n2->g_value + n2->h_value;// f2 = g2 + h2
  return f1 > f2; 
}

RouteModel::Node *RoutePlanner::NextNode() {
    std::sort(this->open_list.begin(), this->open_list.end(), Compare);
    auto low = this->open_list.back();
    auto * cur = low;
    this->open_list.pop_back();
    return  cur;
}




// TODO 6: Complete the ConstructFinalPath method to return the final path found from your A* search.
// Tips:
// - This method should take the current (final) node as an argument and iteratively follow the 
//   chain of parents of nodes until the starting node is found.
// - For each node in the chain, add the distance from the node to its parent to the distance variable.
// - The returned vector should be in the correct order: the start node should be the first element
//   of the vector, the end node should be the last element.

std::vector<RouteModel::Node> RoutePlanner::ConstructFinalPath(RouteModel::Node *current_node) {
    // Create path_found vector
    distance = 0.0f;
    std::vector<RouteModel::Node> path_found  ;
  int i=0;
     while (current_node!=this->start_node){
        distance += current_node->distance(*current_node->parent);
        path_found.emplace_back(*current_node);
        current_node = current_node->parent;
        i+=1;
        cout<< i << std::endl;
    }
    path_found.push_back(*current_node);
    cout << i+1;
    distance *= m_Model.MetricScale(); // Multiply the distance by the scale of the map to get meters.
    reverse(path_found.begin(), path_found.end());
    return path_found;

  

}


// TODO 7: Write the A* Search algorithm here.
// Tips:
// - Use the AddNeighbors method to add all of the neighbors of the current node to the open_list.
// - Use the NextNode() method to sort the open_list and return the next node.
// - When the search has reached the end_node, use the ConstructFinalPath method to return the final path that was found.
// - Store the final path in the m_Model.path attribute before the method exits. This path will then be displayed on the map tile.



    // TODO: Implement your solution here.
  void RoutePlanner::AStarSearch() {
    RouteModel::Node *end_node = this->end_node;
    RouteModel::Node *current = this->start_node;
    this->AddNeighbors(current);
    while (this->open_list.size() > 0 )  {
        current = this->NextNode();
        if (current == end_node) {
          cout << "end!" << std::endl;
             auto final_path = this->ConstructFinalPath(current);
             m_Model.path = final_path;

            break;
        }
        cout << "potas" << std::endl;
          this->AddNeighbors(current);
    }    



    }

