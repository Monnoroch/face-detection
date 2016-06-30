FROM ubuntu:14.04

ENV HOME_PREFIX /root
ENV GOROOT $HOME_PREFIX/go
ENV GOPATH $HOME_PREFIX/dev/go
ENV PATH $HOME_PREFIX/anaconda/bin:$GOROOT/bin:$PATH
ENV LANG en_US.UTF-8
ENV LC_MESSAGES POSIX

COPY .ssh $HOME_PREFIX/.ssh

RUN \
	locale-gen en_US.UTF-8 && \
	update-locale LANG=en_US.UTF-8 LC_MESSAGES=POSIX

# packages
RUN \
	apt-get update -y && \
	apt-get install -y openssh-server wget git build-essential g++ curl mercurial cmake pkg-config \
		libjpeg8-dev libtiff4-dev libjasper-dev libpng12-dev \
		libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
		libgtk2.0-dev libatlas-base-dev gfortran python-dev python3-dev python3-pip

# sshd
RUN \
	mkdir -p /var/run/sshd

# setup
RUN \
	mkdir -p $HOME_PREFIX/tmp && \
	echo "export HOME_PREFIX=$HOME_PREFIX" >> $HOME_PREFIX/.bashrc && \
	echo "export LANG=en_US.UTF-8" >> $HOME_PREFIX/.bashrc && \
        echo "export LC_MESSAGES=POSIX" >> $HOME_PREFIX/.bashrc

# tini
RUN \
	cd $HOME_PREFIX/tmp && \
	TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
	curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
	dpkg -i tini.deb && \
	rm tini.deb && \
	apt-get clean

# go
RUN \
	cd $HOME_PREFIX/tmp && \
	mkdir -p $GOROOT && \
        mkdir -p $GOPATH && \
        mkdir -p $GOPATH/src && \
        wget https://storage.googleapis.com/golang/go1.5.1.linux-amd64.tar.gz && \
        tar -C $GOROOT/.. -xzf go1.5.1.linux-amd64.tar.gz && \
        rm go1.5.1.linux-amd64.tar.gz && \
        echo "export GOROOT=$HOME_PREFIX/go" >> $HOME_PREFIX/.bashrc && \
        echo "export GOPATH=$HOME_PREFIX/dev/go" >> $HOME_PREFIX/.bashrc && \
        echo "export PATH=$GOROOT/bin:$PATH" >> $HOME_PREFIX/.bashrc && \
        go get code.google.com/p/go.net/websocket && \
        go get github.com/siddontang/ledisdb/config && \
        go get github.com/siddontang/ledisdb/ledis

# anaconda
RUN \
	cd $HOME_PREFIX/tmp && \
	wget https://3230d63b5fc54e62148e-c95ac804525aac4b6dba79b00b39d1d3.ssl.cf1.rackcdn.com/Anaconda3-2.4.0-Linux-x86_64.sh && \
	/bin/bash ./Anaconda3-2.4.0-Linux-x86_64.sh -b -p $HOME_PREFIX/anaconda && \
	echo "export PATH=$HOME_PREFIX/anaconda/bin:$PATH" >> $HOME_PREFIX/.bashrc && \
	rm ./Anaconda3-2.4.0-Linux-x86_64.sh

# NN python packages
RUN \
	cd $HOME_PREFIX/tmp && \
	pip install theano lasagne && \
	pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git && \
	pip install --upgrade --no-deps git+git://github.com/Lasagne/Lasagne.git && \
	pip install --upgrade requests && \
	git clone git://github.com/lisa-lab/pylearn2.git && \
	cd pylearn2 && \
	python setup.py develop && \
	cd $HOME_PREFIX/tmp && \
	rm -rf pylearn2

# opencv
RUN \
	cd $HOME_PREFIX/tmp && \
	git clone https://github.com/Itseez/opencv.git && \
	git clone https://github.com/Itseez/opencv_contrib.git && \
	cd opencv_contrib && \
	git checkout 3.0.0 && \
	cd ../opencv && \
	git checkout 3.0.0 && \
	mkdir build && \
	cd build && \
	/usr/bin/pip3 install numpy && \
	cmake -D CMAKE_BUILD_TYPE=RELEASE \
		-D CMAKE_INSTALL_PREFIX=/usr/local \
		-D INSTALL_C_EXAMPLES=ON \
		-D INSTALL_PYTHON_EXAMPLES=ON \
		-D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
		-D BUILD_EXAMPLES=ON .. && \
	make -j4 && \
	make install && \
	ldconfig && \
	ln -s /usr/local/lib/python3.4/dist-packages/cv2.cpython-34m.so $HOME_PREFIX/anaconda/lib/python3.5/site-packages/cv2.so && \
	cd $HOME_PREFIX/tmp && \
	rm -rf opencv_contrib opencv

# xgboost
RUN \
	cd $HOME_PREFIX/tmp && \
	git clone https://github.com/dmlc/xgboost.git && \
	cd xgboost && \
	./build.sh && \
	cd python-package && \
	python setup.py install && \
	cd $HOME_PREFIX/tmp && \
	rm -rf xgboost