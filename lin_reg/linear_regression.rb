require 'matrix'
require_relative '../utils'

class LinearRegression
  MAX_ITERS = 5000

  def initialize(x_mat = nil, y_vec = nil)
    @x_mat = x_mat
    @y_vec = y_vec
    @num_instances = 0
    @num_features = 0
    @theta_vec = nil    
  end

  def debug_print
    puts "\nX:"
    puts @x_mat.inspect
    puts "\ny:"
    puts @y_vec.inspect
    puts "\nWeights: "
    puts @theta_vec.inspect
    puts "\nHypothesis: "
    puts h(@theta_vec)
    puts "\nJ = " + J(@theta_vec).to_s
  end

  def load_data(data_file)

  end

  def load_model(model)

  end

  def train(alpha = 0.01)
    @num_instances = @y_vec.size
    @x_mat = @x_mat.normalize_columns
    @x_mat = @x_mat.insert_column(0, Vector.ones(@num_instances))
    @num_features = @x_mat.column_size
    @theta_vec = grad_desc(Vector.zeros(@num_features), alpha)
  end

  # instance is a Vector of row
  def predict(instance)
    
  end

  def cost
    J(@theta_vec)
  end
  
  private

  def h(theta_vec)
    @x_mat * theta_vec
  end

  def diff_J(theta_vec, j)
    x_j_vec = @x_mat.column(j)
    ( x_j_vec.transpose * (h(theta_vec) - @y_vec) )/@num_instances
  end

  def new_thetas(theta_vec, alpha)
    new_theta_col = []
    theta_vec.to_a.each_with_index do |theta_j, j|
      new_theta_col << theta_j - (alpha * diff_J(theta_vec, j))[0]
    end
    Vector.elements(new_theta_col)
  end

  def grad_desc(theta_vec, alpha)
    MAX_ITERS.times do |i|
      j0 = J(theta_vec)
      theta_vec_new = new_thetas(theta_vec, alpha)
      j1 = J(theta_vec_new)

      #alpha is too large
      raise(ArgumentError, "Alpha: #{alpha} is too large") if j1 > j0 && i < 1 && j0 > 10
      
      #covergence has been reached
      if ( j1 > j0 && i > 1 ) ||  #The cost has reached the rigth side of U
        ( (j0 - j1) <= 0.001 )    #The cost reduction is too little to matter
        return theta_vec
      end
      
      theta_vec = theta_vec_new
    end
    return theta_vec
  end

  def J(theta_vec)
    d_mat = h(theta_vec) - @y_vec
    ((d_mat.transpose)*d_mat)[0]/(2*@num_instances)
  end

end