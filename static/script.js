function validateEmail(email) {
    const re = /^(([^<>()[\]\\.,;:\s@"]+(\.[^<>()[\]\\.,;:\s@"]+)*)|(".+"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))$/;
    return re.test(String(email).toLowerCase());
  }
  let z1 = false
  let x1 = document.getElementById('name');
  x1.addEventListener('focusout', function(event) {
    z1 = true;
    let y1 = document.getElementById('ne');
    if (x1.value === '' || x1.value ==null) {
      y1.innerHTML = "Name field is required";
      x1.classList.replace('formdeco', 'formdeco1');
    }
    else if (x1.value.length < 3) {
      x1.classList.replace('formdeco', 'formdeco1');
      y1.innerHTML = "Invalid name"
    }
    else {
      x1.classList.replace('formdeco1', 'formdeco');
      y1.innerHTML = "";
      z1 = true;
    }
  });

  z2 = false;
  let x2 = document.getElementById('email');
  x2.addEventListener('focusout', function(event) {
    let y2 = document.getElementById('ee');
    let email = x2.value.trim();
    if (x2.value === '' || x2.value ==null) {
      y2.innerHTML = "Email field is required";
      x2.classList.replace('formdeco', 'formdeco1');
    }
    else if (!validateEmail(email)) {
      y2.innerHTML = "Invalid Email";
      x2.classList.replace('formdeco', 'formdeco1');
    }
    else {
      x2.classList.replace('formdeco1', 'formdeco');
      y2.innerHTML = "";
      z2 = true;
    }
  });

  z3 = false;
  let x3 = document.getElementById('message');
  x3.addEventListener('focusout', function(event) {
    let y3 = document.getElementById('me');
    if (x3.value === '' || x3.value ==null) {
      y3.innerHTML = "Message field is required";
      x3.classList.replace('formdeco', 'formdeco1');
    }
    else {
      x3.classList.replace('formdeco1', 'formdeco');
      y3.innerHTML = "";
      z3 = true;
    }
  });

  function check() {
    if (z1 && z2 && z3) {
      return true;
    }
    else {
      return false;
    }
  }