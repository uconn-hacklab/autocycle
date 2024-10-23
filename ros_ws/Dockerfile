FROM ros:humble
ARG USERNAME=USERNAME
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN if id -u $USER_UID ; then userdel `id -un $USER_UID` ; fi

# prevent interactive messages in apt install
ARG DEBIAN_FRONTEND=noninteractive

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && echo "source /opt/ros/humble/setup.bash" >> /home/${USERNAME}/.profile
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y python3-pip
ENV SHELL /bin/bash

# install development tools
RUN apt-get update \
    && apt-get install -q -y --no-install-recommends \
        apt-utils \
        ccache \
        clang \
        cmake \
        git \
        lld \
        llvm \
        vim \
        nano \
        libglfw3-dev \
        python3-colcon-mixin \
        python3-colcon-common-extensions \
        python3-colcon-lcov-result \
        python3-colcon-coveragepy-result \
        python3-colcon-mixin \
        python3-pip \
        python3-rosdep \
        python3-vcstool \
        wget \
        bash-completion \
    && rm -rf /var/lib/apt/lists/*

# install important ROS dependencies
RUN apt-get update \
    && apt-get install -q -y --no-install-recommends \
        ros-humble-ur \
        ros-humble-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get upgrade -y

USER $USERNAME

RUN echo "source /opt/ros/humble/setup.bash" >> /home/${USERNAME}/.bashrc \
 && echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >> /home/${USERNAME}/.bashrc \
 && echo "source /etc/bash_completion" >> /home/${USERNAME}/.bashrc \
 && echo "source /usr/share/bash-completion/completions/git" >> /home/${USERNAME}/.bashrc


CMD ["/bin/bash"]
