# deep-fourier-net
\bold{Absract}

A generative model models the generation of data in the real world. It explains how data
is created in terms of a probabilistic model. Image editing, visual domain adaptation, data
augmentation for discriminative models, and assisted creative production are all examples of
generative models.

To be specific, datasets often have fewer data points in some sections of
their domain, which might reduce model performance if not handled appropriately. Generative
models may be used to modify datasets and upsample low-density areas. This is particularly
beneficial for skewed datasets and simulated data settings. In addition to that, in a wide range
of mathematics and engineering areas, high-dimensional probability distributions are crucial
elements. As in many domains, such as visual data, most distribution content is constrained
to a small region of space, we can safely compress it to a space with a lower dimension. A great
test of our capacity to represent and work with high-dimensional probability distributions is
the training and sampling of generative models that use the same approach. While significant
progress has been achieved, there are still issues that must be addressed. For example, 
high-quality synthesis for complicated scenes or multi-class datasets is still a long way off.
There is a lack of universal objective criteria for assessing the quality of produced samples.
While image production has delivered remarkable results, there is still a significant possibility
for advancement in new areas such as cross-modal generation and video generation. In
this work, we are trying to propose a new network in order to explicitly incorporate the
frequency representation in the process of image generation in order to create a stable novel
image generative model. The backbone of our model is a composition of low-frequency
low-dimensional structures, which is designed to store high-dimensional images in a lowdimensional
space in the frequency domain. If all images in a dataset could be assigned to
separate points in our proposed latent space, we could get new samples from this space to
generate fake images. We have two main contributions in this thesis:
• Introducing a unique Fourier-based algorithm for the lossy image compression task with
an adjustable compression rate.
• Proposing a novel image generative model which relies on the same idea behind the
compression task.
