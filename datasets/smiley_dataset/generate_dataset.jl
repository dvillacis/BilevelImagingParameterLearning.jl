
# Generating synthetic dataset using gaussian noise on the smiley image

using Images

original_path = "smiley.png"

# Load original image
original_img = load(original_path)
Float64.(original_img)

# Noise levels
noise_levels = [0.1,0.2,0.4,0.9]

# Writing images
for i = 1:length(noise_levels)
    n = noise_levels[i]
    n_img = original_img + n * randn(size(original_img))
    n_img[n_img .< 0] .= 0.
    n_img[n_img .> 1] .= 1.
    name_out =  string(i,"_noisy_",n,"_",original_path)
    save(name_out,n_img)
    println(string("saved ",name_out))
end
