#camera_calibration
import cv2 as cv
import numpy as np
import os
import pathlib
import glob
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
img_dir1   = 'C:/Users/sesa564053/Desktop/Yandex_course/Object_Recognition/1/'
img_dir2  = 'C:/Users/sesa564053/Desktop/Yandex_course/Object_Recognition/2/'
grid_size = (9,6)
square_size = (11,11)
img_type = '*.jpg'

def calibration(img_dir,img_type,grid_size,square_size):
		
	objp = np.zeros((grid_size[0]*grid_size[1],3), np.float32)
	objp[:,:2] = np.mgrid[0:grid_size[0],0:grid_size[1]].T.reshape(-1,2)
	objpoints = []
	imgpoints = []
	images = glob.glob(img_dir+'*'+img_type)
	for img_nm in images:
		#print(img_nm)
		img = cv.imread(img_nm)
		
		gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
		ret, corners = cv.findChessboardCorners(gray,(grid_size[0],grid_size[1]),None)
		print(ret)
		if ret == True:
			objpoints.append(objp)
			corners2 = cv.cornerSubPix(gray,corners,square_size,(-1,-1),criteria)
			imgpoints.append(corners)
			cv.drawChessboardCorners(img,grid_size,corners2,ret)
			cv.imshow('img',img)
			cv.waitKey(500)
		cv.destroyAllWindows()
		ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
		h,  w = img.shape[:2]
		newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
		print(newcameramtx,roi)
		dst = cv.undistort(img, mtx, dist, None, newcameramtx)
		# crop the image
		x, y, w, h = roi
		dst = dst[y:y+h, x:x+w]
		cv.imwrite('calibresult.png', dst)
		error_check(objpoints,imgpoints,rvecs,tvecs,newcameramtx,dist)
		return newcameramtx,dist,rvecs,tvecs,imgpoints,objpoints,gray

def stereo_calibration():
	flags = 0
	flags |= cv.CALIB_FIX_INTRINSIC
	# flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
	flags |= cv.CALIB_USE_INTRINSIC_GUESS
	flags |= cv.CALIB_FIX_FOCAL_LENGTH
	# flags |= cv2.CALIB_FIX_ASPECT_RATIO
	flags |= cv.CALIB_ZERO_TANGENT_DIST
	# flags |= cv2.CALIB_RATIONAL_MODEL
	# flags |= cv2.CALIB_SAME_FOCAL_LENGTH
	# flags |= cv2.CALIB_FIX_K3
	# flags |= cv2.CALIB_FIX_K4
	# flags |= cv2.CALIB_FIX_K5
	mtx_l, dist_l, rvecs_l,tvecs_l,imgpnts_l,objpnts_l,gray_l = calibration(img_dir1, img_type,grid_size,square_size)
	mtx_r, dist_r, rvecs_r,tvecs_r,imgpnts_r,objpnts_r, gray_r = calibration(img_dir1, img_type,grid_size,square_size)
	stereocalib_criteria = (cv.TERM_CRITERIA_MAX_ITER +
								cv.TERM_CRITERIA_EPS, 100, 1e-5)
	#print(np.array_equal(objpnts_l,objpnts_r))
	ret, M1, d1, M2, d2, R, T, E, F = cv.stereoCalibrate(
			objpnts_l, imgpnts_l,
			imgpnts_r, mtx_l, dist_l, mtx_r,
			dist_r, gray_l.shape[::-1],
			criteria=stereocalib_criteria, flags=flags)
	with open ('output_params.txt','w') as file:
		file.write('Intrinsic_matrix_cam1:\n' + str(M1) +'\n')
		file.write('Distortion_index_cam1:\n'+str(d1)+'\n')
		file.write('Intrinsic_matrix_cam2:\n'+str(M2)+'\n')
		file.write('Distortion_index_cam2:\n'+str(d2)+'\n')
		file.write('R vector of 2 cams:\n'+str(R)+'\n')
		file.write('Translation vector of 2 cams:\n'+str(T)+'\n')
		file.write('Essential matrix:\n' + str(E)+'\n')
		file.write('F:\n'+str(F)+'\n')
		print('Intrinsic_mtx_1', M1)
		print('dist_1', d1)
		print('Intrinsic_mtx_2', M2)
		print('dist_2', d2)
		print('R', R)
		print('T', T)
		print('E', E)
		print('F', F)

	rectify_scale = 1

	rect_l, rect_r, proj_mat_l, proj_mat_r,Q,roiL,roiR = cv.stereoRectify(M1,d1,M2,d2,
																		gray_l.shape[::-1],R,T,rectify_scale,(0,0))
	
	#return M1, d1, M2, d2, R, T, E, F
'''
def stereo_rectify():
	M1, d1, M2, R, T, E, F = stereo_calibration()
'''
	
def error_check(objpoints,imgpoints,rvecs,tvecs,mtx,dist):
	mean_error = 0
	print(imgpoints)
	for i in range(len(objpoints)):
		imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
		error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
		mean_error += error
	print( "total error: {}".format(mean_error/len(objpoints)) )

if __name__ == "__main__":
	stereo_calibration()


	