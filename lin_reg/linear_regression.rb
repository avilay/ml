require 'matrix'
require_relative '../utils'
require_relative 'linear_regression_model'

class LinearRegressionAnalytics
  attr_accessor :num_iters, :cost_history, :alpha

  def initialize
    @cost_history = []
  end
end

class LinearRegression
  MAX_ITERS = 5000

  attr_reader :analytics
  attr_accessor :x_mat, :y_vec, :model

  def initialize(x_mat = nil, y_vec = nil)
    @x_mat = x_mat
    @y_vec = y_vec
    @model = nil
    @analytics = nil   
  end

  def debug_print
    if @x_mat && @y_vec && @model
      puts "\nX:"
      puts @x_mat.inspect
      puts "\ny:"
      puts @y_vec.inspect    
      puts "\nHypothesis: "
      puts h(@model.theta_vec)
      puts "\ncost = " + cost.to_s
    else
      puts "\nX: not set"
      puts "\ny: not set"
      puts "\nHypothesis: not set"      
      puts "\ncost = not available"
    end

    if @model
      puts "\nNormal Params:"
      @model.normal_params.each do |np|
        if np
          puts np.to_h.to_s
        else
          puts 'nil'
        end
      end
      puts "\nWeights: "
      puts @model.theta_vec.inspect  
    else
      puts "\nNormal Params: not set"
      puts "\nWeights: not set"
    end

    if @analytics
      puts "\nNumber of iterations to convergence:"
      puts @analytics.num_iters.to_s
      puts "\nAlpha:"
      puts @analytics.alpha.to_s
      puts "\nCost history:"
      puts @analytics.cost_history.inspect
    end
  end
  
  def prepare_data
    @x_mat = @x_mat.normalize_columns
    @x_mat = @x_mat.insert_column(0, Vector.ones(@x_mat.row_size))    
  end

  def train(alpha = 0.01)
    @num_instances = @y_vec.size
    @num_features = @x_mat.column_size
    @model = LinearRegressionModel.new
    @analytics = LinearRegressionAnalytics.new
    @model.theta_vec = grad_desc(Vector.zeros(@num_features), alpha)
    @model.normal_params = @x_mat.normal_params
    @analytics.alpha = alpha
  end

  def predict(instance)
    if instance.instance_of?(Vector) || instance.instance_of?(Array)
      predict_vector(instance)
    elsif instance.instance_of?(Matrix)
      predict_matrix(instance)
    else
      raise "instance has to be of type Matrix or Vector or Array."
    end
  end

  def cost
    if @x_mat then J(@model.theta_vec) else nil end
  end

  # **************************************** #
  private

  def predict_matrix(data_mat)
    @num_instances = data_mat.row_size
    data_mat = data_mat.insert_column(0, Vector.ones(@num_instances))
    cols = data_mat.column_vectors
    cols.each_with_index do |col, i|
      next unless @model.normal_params[i]
      avg = @model.normal_params[i].avg
      stdev = @model.normal_params[i].stdev
      cols[i] = col.to_a.map{|x| (x - avg)/stdev}      
    end
    @x_mat = Matrix.columns(cols)
    @num_features = @x_mat.column_size  
    return h(@model.theta_vec)
  end

  def predict_vector(data_vec)
    data_vec = data_vec.to_a
    data_vec.unshift(1)
    data_vec.each_with_index do |x, i|
      next unless @model.normal_params[i]
      data_vec[i] = (x - @model.normal_params[i].avg)/@model.normal_params[i].stdev
    end
    x_vec = Vector.elements(data_vec)
    ret = @model.theta_vec.transpose * x_vec
    return ret[0]
  end

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
      @analytics.cost_history << j1
      #alpha is too large
      raise(ArgumentError, "Alpha: #{alpha} is too large") if j1 > j0 && i < 1 && j0 > 10
      
      #covergence has been reached
      if ( j1 > j0 && i > 1 ) ||  #The cost has reached the rigth side of U
        ( (j0 - j1) <= 0.001 )    #The cost reduction is too little to matter
        @analytics.num_iters = i
        return theta_vec
      end
      
      theta_vec = theta_vec_new
    end
    @analytics.num_iters = MAX_ITERS
    return theta_vec
  end

  def J(theta_vec)
    d_mat = h(theta_vec) - @y_vec
    ((d_mat.transpose)*d_mat)[0]/(2*@num_instances)
  end

end