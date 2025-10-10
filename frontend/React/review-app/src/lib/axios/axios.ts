import axios from "axios";

const API_BASE: string =
 "/api";

const client = axios.create({
  baseURL: API_BASE,
  withCredentials: true,
});

export default client;
