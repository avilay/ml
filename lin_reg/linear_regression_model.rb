class LinearRegressionModel
  attr_accessor :theta_vec, :normal_params

  def initialize(params = nil)
    if params
      @theta_vec = params[:theta_vec]
      @normal_params = params[:normal_params]
    end
  end

  def save(filename)
    
  end

  def load(filename)
    
  end
end